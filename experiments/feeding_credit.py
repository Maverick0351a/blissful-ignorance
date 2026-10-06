"""Read-only instrumentation of recorded experimental PPO training.

The original continuation trajectory is reproduced, never extended. The native
learner supplies every action and update. Diagnostic probes never enter a rollout.
"""
import argparse
import copy
from dataclasses import asdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from agents.affordances import action_mask
from agents.network import ACTIONS, encode_observation
from agents.sequence_ppo import Config, advantages
from experiments.category_trial import (MeasuredCategory, population_for, check_budget,
    memory_usage, weight_hashes, bodies, write, sha)
from experiments.near_food_trial import make_world
from sim.world import World

EAT = ACTIONS.index({'verb': 'eat'})
GATHER = ACTIONS.index({'verb': 'gather'})
TONE = [i for i, a in enumerate(ACTIONS) if a['verb'] == 'tone']
PROBE_BASE = 80400000


def fingerprint(value):
    """Hash the full nested learning state, including optimizers and RNGs."""
    h = hashlib.sha256()
    def visit(v):
        if isinstance(v, torch.Tensor):
            h.update(f'tensor:{v.dtype}:{tuple(v.shape)}:'.encode())
            h.update(v.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(v, dict):
            h.update(b'dict{')
            for key in sorted(v, key=lambda x: (type(x).__name__, repr(x))):
                visit(key); visit(v[key])
            h.update(b'}')
        elif isinstance(v, (tuple, list)):
            h.update((type(v).__name__+'[').encode())
            for item in v:visit(item)
            h.update(b']')
        else:h.update((type(v).__name__+':'+json.dumps(v, allow_nan=False)+';').encode())
    visit(value)
    return h.hexdigest()


def tensor_probe(observation, hidden):
    patch, features = encode_observation(observation)
    return dict(patch=patch, features=features,
                mask=torch.tensor([action_mask(observation, ACTIONS)]),
                hidden=tuple(t.detach().clone() for t in hidden))


def probe(brain, entry, zero_context=True):
    hidden = brain.model.initial_state() if zero_context else entry['hidden']
    with torch.no_grad():
        logits, _, _ = brain.model(entry['patch'], entry['features'], hidden)
        p = brain.distribution(logits, entry['mask']).probs[0]
    return dict(eat=float(p[EAT]), gather=float(p[GATHER]), tone=float(p[TONE].sum()))


def canonical_inputs(brain, rid):
    return {f'{fixture}-{case}':tensor_probe(make_world(PROBE_BASE+case, case, fixture)[0].observe(rid),
                                            brain.model.initial_state())
            for fixture in ('adjacent', 'carried') for case in range(4)}


def sequence_probabilities(brain, rows):
    with torch.no_grad():
        logits, _ = brain._sequence(torch.cat([r['patch'] for r in rows]),
                                   torch.cat([r['features'] for r in rows]), rows[0]['hidden'])
        return brain.distribution(logits, torch.cat([r['mask'] for r in rows])).probs


def credit_terms(rows, journal, bootstrap, config):
    if len(rows) != len(journal):raise ValueError('Reward journal does not cover the rollout')
    values = torch.tensor([r['value'] for r in rows])
    terminals = [r['terminal'] for r in rows]
    adv, returns = advantages([r['reward'] for r in rows], values, bootstrap,
                              terminals, config.gamma, config.gae_lambda)
    normalized = (adv-adv.mean())/(adv.std(unbiased=False)+1e-8)
    parts = {}
    keys = sorted(set().union(*(j['reward'] for j in journal)))
    for key in keys:
        parts[key], _ = advantages([j['reward'].get(key, 0.) for j in journal],
                                  torch.zeros_like(values), 0., terminals, config.gamma, config.gae_lambda)
    parts['value_residual'], _ = advantages([0.]*len(rows), values, bootstrap,
                                           terminals, config.gamma, config.gae_lambda)
    if not torch.allclose(sum(parts.values()), adv, atol=2e-5, rtol=2e-5):
        raise AssertionError('Advantage decomposition failed')
    return adv, normalized, returns, parts


def gradient_components(brain, rows, normalized, returns):
    """First-epoch gradient measurements only; autograd.grad never writes .grad."""
    logits, predicted = brain._sequence(torch.cat([r['patch'] for r in rows]),
                                       torch.cat([r['features'] for r in rows]), rows[0]['hidden'])
    d = brain.distribution(logits, torch.cat([r['mask'] for r in rows]))
    selected = torch.tensor([r['chosen'] for r in rows])
    ratio = (d.log_prob(selected)-torch.tensor([r['logp'] for r in rows])).exp()
    losses = dict(actor=-torch.minimum(ratio*normalized,
        ratio.clamp(1-brain.config.clip, 1+brain.config.clip)*normalized).mean(),
        value=.5*(predicted-returns).square().mean(),
        entropy=-brain.config.entropy*d.entropy().mean())
    parameters = tuple(brain.model.parameters())
    vectors = {}
    for name, loss in losses.items():
        grads = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
        vectors[name] = torch.cat([(g if g is not None else torch.zeros_like(p)).flatten()
                                  for p, g in zip(parameters, grads)]).detach()
    norms = {name:float(v.norm()) for name,v in vectors.items()}
    total = float(sum(vectors.values()).norm())
    cosine = float(torch.dot(vectors['actor'], vectors['value']))/max(1e-20,norms['actor']*norms['value'])
    return dict(losses={k:float(v.detach()) for k,v in losses.items()}, norms=norms,
                combined_norm=total, clipping_scale=min(1., 1./(total+1e-6)),
                actor_value_cosine=cosine,
                interpretation='First epoch before Adam; norms do not establish a causal bottleneck')


class CreditBrain(MeasuredCategory):
    def __init__(self, seed, config):
        super().__init__(seed, config)
        self.records = []; self.anchor = None; self.context = {}; self.probes = {}

    def _learn(self, bootstrap):
        if not self.buffer:return
        before_state = fingerprint(self.state())
        rng_before = torch.random.get_rng_state().clone()
        gradients_before = [None if p.grad is None else p.grad.clone() for p in self.model.parameters()]
        rows = self.buffer; journal = self.journal[-len(rows):]
        if len(rows) != self.config.rollout:raise ValueError('Diagnostic requires complete recorded rollouts')
        adv, normalized, returns, parts = credit_terms(rows, journal, bootstrap, self.config)
        before = sequence_probabilities(self, rows)
        meal_indices = [i for i,r in enumerate(rows) if r['chosen']==EAT and journal[i]['reward'].get('food',0)>0]
        first_anchor = self.anchor is None and bool(meal_indices)
        if first_anchor:
            i = meal_indices[0]
            self.anchor = {k:copy.deepcopy(rows[i][k]) for k in ('patch','features','mask','hidden')}
            self.anchor.update(decision=journal[i]['decision'], case=self.context['case'], row=i,
                               fullness=float(rows[i]['body'][0])*25)
        canonical_before = {k:probe(self,v) for k,v in self.probes.items()}
        anchor_before = None if self.anchor is None else {
            'zero_context':probe(self,self.anchor), 'fixed_context':probe(self,self.anchor,False)}
        gradient = gradient_components(self, rows, normalized, returns)
        if fingerprint(self.state()) != before_state or not torch.equal(rng_before, torch.random.get_rng_state()):
            raise AssertionError('Diagnostics mutated state or Torch RNG')
        for old,p in zip(gradients_before,self.model.parameters()):
            if (old is None) != (p.grad is None) or old is not None and not torch.equal(old,p.grad):
                raise AssertionError('Diagnostics mutated gradient buffers')
        # The production update itself remains unchanged.
        super()._learn(bootstrap)
        after = sequence_probabilities(self, rows)
        transitions = []
        for i,(r,j) in enumerate(zip(rows,journal)):
            choice = r['chosen']
            if abs(float(before[i,choice])-math.exp(r['logp'])) > 2e-6:
                raise AssertionError('Sequence replay does not match behavior probabilities')
            if abs(sum(j['reward'].values())-r['reward']) > 1e-9:
                raise AssertionError('Reward components do not match learner reward')
            transitions.append(dict(index=i,decision=j['decision'],action=dict(ACTIONS[choice]),
                reward=r['reward'],reward_parts=j['reward'],value=r['value'],terminal=r['terminal'],
                advantage=float(adv[i]),normalized_advantage=float(normalized[i]),target_return=float(returns[i]),
                components={k:float(v[i]) for k,v in parts.items()},
                ordinary_meal=i in meal_indices,fullness=float(r['body'][0])*25,
                chosen_probability_before=float(before[i,choice]),chosen_probability_after=float(after[i,choice]),
                eat_probability_before=float(before[i,EAT]),eat_probability_after=float(after[i,EAT])))
        self.records.append(dict(**self.context,update=self.updates,bootstrap=bootstrap,
            state_before=before_state,state_after=fingerprint(self.state()),
            transitions=transitions,gradient=gradient,learner_diagnostics=copy.deepcopy(self.diagnostics),
            canonical_before=canonical_before,canonical_after={k:probe(self,v) for k,v in self.probes.items()},
            anchor_first=first_anchor,anchor_identity=None if self.anchor is None else {
                k:self.anchor[k] for k in ('decision','case','row','fullness')},
            anchor_before=anchor_before,anchor_after=None if self.anchor is None else {
                'zero_context':probe(self,self.anchor),'fixed_context':probe(self,self.anchor,False)}))


def summarize(records):
    mean = lambda xs: sum(xs)/len(xs) if xs else None
    candidates = {}
    for cid in sorted({r['candidate'] for r in records}):
        updates = [r for r in records if r['candidate']==cid]
        transitions = [t for r in updates for t in r['transitions']]
        meals = [t for t in transitions if t['ordinary_meal']]
        anchor = next((r for r in updates if r['anchor_first']), None)
        retention = {}
        if anchor:
            for context in ('zero_context','fixed_context'):
                base=anchor['anchor_before'][context]['eat']; immediate=anchor['anchor_after'][context]['eat']
                final=updates[-1]['anchor_after'][context]['eat']; later=updates[-1]['update']-anchor['update']
                eligible = immediate-base>1e-6 and later>=4
                retention[context]=dict(before=base,immediate=immediate,final=final,later_updates=later,
                    eligible=eligible,retains_half_gain=None if not eligible else final-base >= .5*(immediate-base))
        by_verb = {}
        for verb in sorted({t['action']['verb'] for t in transitions}):
            ts=[t for t in transitions if t['action']['verb']==verb]
            by_verb[verb]=dict(decisions=len(ts),positive_advantage=sum(t['normalized_advantage']>0 for t in ts),
                mean_normalized_advantage=mean([t['normalized_advantage'] for t in ts]),
                positive_food_credit_sum=sum(max(0,t['components'].get('food',0)) for t in ts))
        canonical = {}
        for fixture,verb in (('carried','eat'),('adjacent','gather')):
            canonical[fixture]=dict(before=mean([v[verb] for k,v in updates[0]['canonical_before'].items() if k.startswith(fixture)]),
                after=mean([v[verb] for k,v in updates[-1]['canonical_after'].items() if k.startswith(fixture)]))
        candidates[cid]=dict(updates=len(updates),ordinary_meals=len(meals),
            positive_meal_advantages=sum(t['normalized_advantage']>0 for t in meals),
            reinforced_meals=sum(t['eat_probability_after']-t['eat_probability_before']>1e-6 for t in meals),
            mean_meal_probability_change=mean([t['eat_probability_after']-t['eat_probability_before'] for t in meals]),
            canonical=canonical,retention=retention,actions=by_verb,
            mean_gradient_norms={k:mean([r['gradient']['norms'][k] for r in updates]) for k in ('actor','value','entropy')})
    return dict(candidates=candidates,updates=len(records),ordinary_meals=sum(c['ordinary_meals'] for c in candidates.values()),
        positive_meal_advantages=sum(c['positive_meal_advantages'] for c in candidates.values()),
        reinforced_meals=sum(c['reinforced_meals'] for c in candidates.values()),
        milestone_confirmed=False,
        interpretation='Retrospective diagnostic of one six-candidate experimental pool. No new lives or causal memory claim.')


def validate_source(source):
    manifest=json.loads((source/'preregistration.json').read_text())
    receipt=json.loads((source/'complete.json').read_text()); audit=json.loads((source/'audit.json').read_text())
    if manifest['experiment']!='archived-selective-restart' or manifest['smoke'] or not audit['passed']:
        raise ValueError('Requires the audited full archived culling pilot')
    if sha(source/'preregistration.json')!=(source/'preregistration.sha256').read_text().strip():
        raise ValueError('Source preregistration changed')
    for name,digest in receipt['evidence_hashes'].items():
        if sha(source/name)!=digest:raise ValueError('Source evidence changed: '+name)
    for name,digest in manifest['hashes'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Source changed; preserve and review before replay: '+name)
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'runs/culling-pilot-20261006')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--wall-seconds',type=int,default=600)
    args=parser.parse_args(); source=args.source.resolve();out=args.output.resolve()
    if any(not p.is_relative_to(ROOT/'runs') or p==ROOT/'runs' for p in (source,out)):
        parser.error('Use local experimental run directories')
    if not 1<=args.wall_seconds<=600:parser.error('Wall cap must be 1..600 seconds')
    source_manifest=validate_source(source); out.mkdir(parents=True,exist_ok=False)
    for name in ('continue-initial.pt','continue-trained.pt','preregistration.json','complete.json','audit.json'):
        shutil.copy2(source/name,out/('source-'+name))
    names=set(source_manifest['hashes']) | {'experiments/feeding_credit.py','experiments/audit_feeding_credit.py',
                                         'tests/test_feeding_credit.py'}
    manifest=dict(protocol=1,milestone='GT-01',diagnostic='recorded-feeding-update',source=str(source.relative_to(ROOT)),
        source_trace_sha256=sha(source/'trace.jsonl.gz'),source_complete_sha256=sha(source/'complete.json'),
        source_audit_sha256=sha(source/'audit.json'),config=source_manifest['config'],
        replay_ticks=24576,decisions=12288,updates=96,probes_base=PROBE_BASE,
        wall_seconds=args.wall_seconds,audit_wall_seconds=600,evidence_cap_bytes=512*1024*1024,
        primary='Reproduce all sixteen continuation-training lives of each of six category-PPO copies exactly. Record raw and normalized GAE, additive food/hunger/injury/value terms, pre/post meal action probabilities, and first-epoch gradient norms. No selective success-only update replay.',
        probes='Four adjacent and four carried-food private observations at fixed seeds, zero recurrent context. Never sampled or trained on. First ordinary meal per brain additionally tracked on identical encoded input and zero or held-fixed decision-time context.',
        retention='First-meal anchor chosen by chronology. Report every subsequent update. Half-gain retention descriptive flag only if initial probability gain >1e-6 and at least four later updates. Continued learning, fixed old context and repeated observations do not establish memory capacity or long-term skill.',
        decision='No behavioral pass gate or live deployment. More than 1e-6 probability increase counts as reinforcement. Report all six brains separately. This is retrospective mechanism diagnosis, not preregistered behavioral confirmation.',
        boundaries='No new trajectories, forced learner commands, reward or architecture edits, resident reset, Laya call, live endpoint, model download, external service or publication.',
        hashes={name:sha(ROOT/name) for name in sorted(names)})
    write(out/'preregistration.json',manifest)
    (out/'preregistration.sha256').write_text(sha(out/'preregistration.json'),encoding='utf-8')
    with zipfile.ZipFile(out/'source.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(names):archive.write(ROOT/name,name)
    start=time.perf_counter(); deadline=start+args.wall_seconds
    initial=torch.load(out/'source-continue-initial.pt',weights_only=True,map_location='cpu')
    expected_final=torch.load(out/'source-continue-trained.pt',weights_only=True,map_location='cpu')
    config=Config(**manifest['config']); populations={}; records=[]; episodes=steps=decisions=0
    active=False; final_states={}
    try:
        with gzip.open(source/'trace.jsonl.gz','rt',encoding='utf-8') as stream, gzip.open(
                out/'trace.jsonl.gz','wt',encoding='utf-8',compresslevel=1) as subset:
            for line in stream:
                row=json.loads(line)
                if row['type']=='start':
                    meta=row['metadata'];active=meta['stage']=='training' and meta['arm']=='continue'
                    if not active:continue
                    pair=meta['pair']; world=World.from_dict(row['world'])
                    if pair not in populations:
                        p=population_for(meta['training_seed'],'category',world,config)
                        for i,rid in enumerate(('r0','r1')):
                            cid=f'c{pair*2+i}';saved=initial[cid]
                            b=CreditBrain(saved['seed'],config);b.restore(copy.deepcopy(saved))
                            b.probes=canonical_inputs(b,rid);p.brains[rid]=b
                        populations[pair]=p
                    p=populations[pair];p.world=world;p.metrics={rid:p.new_metrics() for rid in p.brains}
                    for i,(rid,b) in enumerate(p.brains.items()):
                        b.rng=random.Random(source_manifest['bases']['training']*10+meta['case']*6+pair*2+i)
                        b.reward_totals={};b.context=dict(candidate=meta['candidates'][rid],case=meta['case'],
                            episode=row['episode'],training_seed=meta['training_seed'])
                    if {rid:weight_hashes(b) for rid,b in p.brains.items()}!=row['weights']:
                        raise AssertionError('Recorded starting weights differ')
                if not active:continue
                if row['type']=='step':
                    if steps%64==0:subset.flush();check_budget(out,deadline)
                    deciding=p.world.tick%4==0
                    result=p.step()
                    commands={rid:b.last_action if deciding else {'verb':'wait'} for rid,b in p.brains.items()}
                    if commands!=row['commands']:raise AssertionError('Native learner diverged from recorded actions')
                    if json.dumps(result,sort_keys=True)!=json.dumps(row['results'],sort_keys=True):
                        raise AssertionError('Physics diverged')
                    if bodies(p.world)!=row['after']:raise AssertionError('Bodies diverged')
                    for rid,b in p.brains.items():
                        if b.last_reward!=row['rewards'][rid]:raise AssertionError('Rewards diverged')
                        if deciding and json.dumps(b.last_observation,sort_keys=True)!=json.dumps(row['observations'][rid],sort_keys=True):
                            raise AssertionError('Private observations diverged')
                    steps+=1;decisions+=len(p.brains) if deciding else 0
                elif row['type']=='end':
                    for rid,b in p.brains.items():
                        b.finish(p.world.observe(rid),terminal=True)
                        records.extend(b.records);b.records=[]
                    if json.dumps(p.world.to_dict(),sort_keys=True)!=json.dumps(row['world'],sort_keys=True):
                        raise AssertionError('World diverged')
                    if {rid:weight_hashes(b) for rid,b in p.brains.items()}!=row['weights_after']:
                        raise AssertionError('Update did not reproduce recorded weights')
                    episodes+=1;write(out/'updates.partial.json',records)
                    if episodes%8==0:print(json.dumps(dict(episodes=episodes,updates=len(records),seconds=time.perf_counter()-start)),flush=True)
                subset.write(line)
        for pair,p in populations.items():
            for i,(rid,b) in enumerate(p.brains.items()):
                cid=f'c{pair*2+i}';final_states[cid]=copy.deepcopy(b.state())
                if fingerprint(b.state())!=fingerprint(expected_final[cid]):
                    raise AssertionError('Full final learning state mismatch: '+cid)
        if (steps,decisions,len(records),episodes)!=(24576,12288,96,48):raise AssertionError('Incomplete trajectory')
        torch.save(final_states,out/'reproduced-final.pt')
        write(out/'updates.json',records);write(out/'analysis.json',summarize(records))
        check_budget(out,deadline)
        if any(sha(ROOT/name)!=digest for name,digest in manifest['hashes'].items()):raise AssertionError('Source changed')
        write(out/'complete.json',dict(seconds=time.perf_counter()-start,episodes=episodes,replayed_ticks=steps,
            replayed_training_decisions=decisions,updates=len(records),exact_final_states=True,
            memory=memory_usage(),live_population_changed=False,
            evidence_hashes={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
        print(json.dumps(dict(complete=True,seconds=time.perf_counter()-start,summary=summarize(records))),flush=True)
    except Exception as exc:
        write(out/'incomplete.json',dict(error=type(exc).__name__,reason=str(exc),episodes=episodes,
            replayed_ticks=steps,seconds=time.perf_counter()-start));raise


if __name__=='__main__':main()
