"""Offline architecture screen. No live saves, network, or model downloads.

Run --help. Each worker owns an independent process; Laya uses an existing local
OpenVINO adapter supplied explicitly, with frozen weights and private context.
This is a two-action bandit built from World mechanics, not full-game PPO.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim.world import World

KINDS = ('random', 'tabular', 'mlp', 'gru', 'fly', 'laya')
SEEDS = (1031, 2053, 4099)
PHASE = 40
HISTORY = 8


def peak_working_mb():
    if os.name != 'nt':
        return None
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD), ('faults', wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in ('peak', 'working', 'paged_peak',
            'paged', 'nonpaged_peak', 'nonpaged', 'pagefile', 'pagefile_peak')]
    c = Counters(); c.cb = ctypes.sizeof(c)
    kernel = ctypes.WinDLL('kernel32'); kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi = ctypes.WinDLL('psapi')
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(c), c.cb):
        return None
    return round(c.peak / 1024**2, 1)


class Learner:
    def __init__(self, kind, seed, laya=None):
        self.kind, self.rng, self.laya = kind, random.Random(seed), laya
        self.q = [0., 0.]
        self.parameters = 0
        if kind in ('mlp', 'gru', 'fly'):
            import torch
            self.torch = torch
            torch.set_num_threads(1); torch.manual_seed(seed)
            if kind == 'fly':
                # Fixed sparse Kenyon-cell-like codes; only output synapses adapt.
                generator = torch.Generator().manual_seed(seed)
                expansion = torch.randn(2, 256, generator=generator)
                self.codes = torch.zeros(2, 256)
                self.codes.scatter_(1, expansion.topk(16, dim=1).indices, 1.)
                self.weights = torch.zeros(256)
                self.parameters = 256
            elif kind == 'mlp':
                self.model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(HISTORY*4, 32),
                    torch.nn.Tanh(), torch.nn.Linear(32, 2))
            else:
                class Recurrent(torch.nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.core = torch.nn.GRU(4, 24, batch_first=True)
                        self.head = torch.nn.Linear(24, 2)
                    def forward(self, x):
                        _, h = self.core(x)
                        return self.head(h[-1])
                self.model = Recurrent()
            if kind != 'fly':
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=.01)
                self.parameters = sum(p.numel() for p in self.model.parameters())

    def tensor(self, history):
        rows = [[float(a == 0), float(a == 1), reward, 1.] for a, reward in history[-HISTORY:]]
        return self.torch.tensor([[[0., 0., 0., 0.]]*(HISTORY-len(rows)) + rows])

    def choose(self, history):
        if self.kind == 'random':
            return self.rng.randrange(2)
        if self.kind == 'laya':
            state = {'recent_oldest_first': [{'choice': 'AB'[a], 'outcome': r} for a,r in history[-HISTORY:]]}
            question = {'action': {'type': 'choice',
                'instructions': 'Choose food A or B to maximize outcome. Positive is good, negative is bad. Associations can change. Use recent outcomes.',
                'criteria': {'A': 'consume A', 'B': 'consume B'}}}
            internal = self.adapter.to_internal(question['action'])
            # Refuse hidden truncation of the benchmark's evidence.
            seq, _ = self.adapter.build_sequence(self.laya.tok, state, internal, 4096,
                                                 self.laya.cfg.get('head_max_len', 192))
            if len(seq) > self.adapter.L:
                raise ValueError('Laya history would be truncated')
            answer = self.laya.system_one(state, question)['answers']['action']
            probabilities = answer['probabilities']
            if not all(__import__('math').isfinite(x) for x in probabilities.values()):
                raise ValueError('Nonfinite Laya output')
            values = [probabilities['A'], probabilities['B']]
        elif self.kind == 'tabular':
            values = self.q
        elif self.kind == 'fly':
            values = (self.codes @ self.weights / 16).tolist()
        else:
            with self.torch.no_grad():
                values = self.model(self.tensor(history))[0].tolist()
        # Same explicit exploration schedule for all nonrandom candidates.
        if self.rng.random() < .15:
            return self.rng.randrange(2)
        best = max(values)
        return self.rng.choice([i for i,v in enumerate(values) if abs(v-best) < 1e-8])

    def learn(self, history, action, reward):
        if self.kind == 'tabular':
            self.q[action] += .3*(reward-self.q[action])
        elif self.kind == 'fly':
            code = self.codes[action]
            prediction = float(code @ self.weights / 16)
            self.weights += .3*(reward-prediction)*code
        elif self.kind in ('mlp', 'gru'):
            estimate = self.model(self.tensor(history))[0, action]
            loss = (estimate-reward)**2
            self.optimizer.zero_grad(); loss.backward()
            self.torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
            self.optimizer.step()


def worker(args):
    load_start = time.perf_counter(); laya = adapter = None
    if args.kind == 'laya':
        os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TRANSFORMERS_OFFLINE'] = '1'
        spec = importlib.util.spec_from_file_location('existing_laya_adapter', args.laya_adapter)
        adapter = importlib.util.module_from_spec(spec); spec.loader.exec_module(adapter)
        laya = adapter.LayaLite('NPU')
    load_seconds = time.perf_counter()-load_start
    rows = []
    for seed in SEEDS:
        agent = Learner(args.kind, seed, laya); agent.adapter = adapter
        w = World(seed); w.residents = {'player': w.residents['player']}
        body = w.residents['player']; history = []; choices = []; timings = []
        initial_safe = seed % 2
        for trial in range(PHASE*2):
            # Independent food probe; physiological resets are explicit and shared.
            body.food = 50; body.health = 100; body.pain = 0; body.unconscious = False
            body.inventory['food'] = body.inventory['amber_fruit'] = 1
            safe = initial_safe if trial < PHASE else 1-initial_safe
            start = time.perf_counter(); action = agent.choose(history)
            timings.append((time.perf_counter()-start)*1000)
            item = 'food' if action == safe else 'amber_fruit'
            ok, message = w.apply_action('player', {'verb':'eat', 'item':item})
            assert ok, message
            # Consequences from actual mechanics, no hidden safe label in input.
            reward = round((body.food-50)/25 - (100-body.health)/10, 4)
            agent.learn(history, action, reward)
            history.append((action, reward))
            choices.append({'trial': trial, 'choice': 'AB'[action], 'safe': action == safe, 'reward':reward})
        rates = {name: sum(x['safe'] for x in choices[a:b])/(b-a) for name,a,b in
                 [('initial_10',0,10), ('learned_last20',20,40),
                  ('reversal_first10',40,50), ('reversal_last20',60,80)]}
        rows.append({'seed':seed, 'rates':rates, 'choices':choices,
                     'median_decision_ms':statistics.median(timings),
                     'p95_decision_ms':sorted(timings)[int(.95*(len(timings)-1))],
                     'trainable_parameters':agent.parameters})
    result = {'kind':args.kind, 'load_seconds':load_seconds, 'rows':rows,
              'peak_process_working_mb':peak_working_mb(),
              'learning': 'frozen weights with supplied recent context' if args.kind=='laya' else
                          'none' if args.kind=='random' else 'online reward prediction'}
    (Path(args.output)/f'{args.kind}.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}), flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=KINDS)
    parser.add_argument('--laya-adapter')
    parser.add_argument('--output',default=str(ROOT/'runs'/'architecture-trial'))
    args=parser.parse_args(); out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    if args.kind:
        worker(args); return
    if (out/'preregistration.json').exists():
        raise SystemExit('Use a fresh output directory to preserve previous experiments')
    plan={'seeds':SEEDS,'choices_per_phase':PHASE,'history':HISTORY,'epsilon':.15,
          'hypotheses':['Experience learners improve safe choice over random.',
                        'Cue reversal reveals adaptation speed.',
                        'Frozen Laya may use supplied history, not learn weights.'],
          'limits':'Narrow bandit; physiology reset every choice; no movement or live world changes. Different learners and representations; not a capacity-controlled architecture comparison.',
          'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'world_sha256':hashlib.sha256((ROOT/'sim/world.py').read_bytes()).hexdigest()}
    (out/'preregistration.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    statuses={}
    for kind in KINDS:
        if kind=='laya' and not args.laya_adapter:
            statuses[kind]='blocked: no local adapter supplied'; continue
        cmd=[sys.executable,__file__,'--kind',kind,'--output',str(out)]
        if args.laya_adapter: cmd+=['--laya-adapter',args.laya_adapter]
        print('Starting '+kind,flush=True)
        try:
            completed=subprocess.run(cmd,capture_output=True,text=True,timeout=240)
            (out/f'{kind}.log').write_text(completed.stdout+completed.stderr,encoding='utf-8')
            statuses[kind]='complete' if completed.returncode==0 else 'failed; see log'
        except subprocess.TimeoutExpired:
            statuses[kind]='timed out at 240 seconds'
        print(kind+': '+statuses[kind],flush=True)
    (out/'status.json').write_text(json.dumps(statuses,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
