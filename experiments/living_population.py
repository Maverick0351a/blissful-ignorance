"""Independent local learning populations, including the primary playable world."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time
import threading
import os
import hashlib
from datetime import datetime, timezone
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from agents.lifelong import Population
from agents.sequence_ppo import Population as SequencePopulation
from agents.preservation import validate_roster
from server import Runtime,WorldServer
from sim.world import World


def load_population(path, *, all_residents=False):
    import torch
    header=torch.load(path,map_location='cpu',weights_only=True)
    cls=SequencePopulation if header.get('backend')=='recurrent-ppo' else Population
    return cls.load(path, require_all=all_residents)


class LearningRuntime(Runtime):
    def __init__(self,root,seed=5729,backend='recurrent-ppo',stage=None,
                 all_residents=True,migrate_world=False,public_demo=False):
        super().__init__(root,seed=seed,autoload=False,public_demo=public_demo)
        self.all_residents=all_residents
        self.migrate_world=migrate_world
        self.checkpoint=self.runs/'population.pt'
        if self.checkpoint.exists():self.population=load_population(self.checkpoint,all_residents=all_residents)
        else:
            source=self.runs/'autosave.json'
            if not source.exists():source=self.runs/'world.json'
            if migrate_world and source.exists():
                world=World.load(source)  # A damaged save must fail instead of silently resetting it.
                backup=self.runs/('legacy-scripted-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
                backup.mkdir()
                for name in ('autosave.json','world.json'):
                    path=self.runs/name
                    if path.exists():shutil.copy2(path,backup/name)
                self.population=SequencePopulation.from_world(world)
                report=dict(source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                            legacy_backup=str(backup),tick=world.tick,residents=list(self.population.brains),
                            backend='recurrent-ppo',initial_weights='fresh independent initialization',
                            world_preserved=True)
                (self.runs/'controller-transition.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
            elif migrate_world:
                self.world.ecology_enabled=True
                self.population=SequencePopulation.from_world(self.world)
            else:self.population=SequencePopulation(seed) if backend=='recurrent-ppo' else Population(seed)
            if stage:
                if backend!='recurrent-ppo':raise ValueError('Development stages require the sequence population')
                traveler=self.population.world.residents['player']
                if stage=='scarcity':
                    from experiments.scarcity_routes import configure
                    arenas=configure(self.population,seed)
                    # Interactive visitors are absent from the formal experiment.
                    self.population.world.residents['player']=traveler
                    traveler.x,traveler.y=arenas['r0'].point(-1,-1)
                else:
                    from experiments.development import configure
                    configure(self.population,seed,stage)
                    traveler.x=23;traveler.y=32  # Walkable observation point in the first room.
                # Fresh experimental rooms contain only their learners and visitor.
                self.population.world.residents={rid:a for rid,a in self.population.world.residents.items()
                                                 if rid in self.population.brains or rid=='player'}
        if all_residents:
            if self.checkpoint.exists() and (set(self.population.brains)!=set(self.population.world.residents)-{'player'}
                                             or getattr(self.population,'neighbors_scripted',True)):
                backup=self.runs/'before-all-residents-population.pt'
                if not backup.exists():shutil.copy2(self.checkpoint,backup)
            self.population.enable_all_residents()
        self.world=self.population.world
        for brain in self.population.brains.values():
            if getattr(brain,'backend',None)=='laya-npu':brain._load_engine()
        self.controller_error=None
        self.manual_pause=self.paused=True
        self.pause_reason='Independent learners ready; paused by you'
        if migrate_world and not self.checkpoint.exists():self._save('autosave.json')

    def _tick(self):
        command=self.pending_action
        try:
            if self.all_residents and set(self.population.brains)!=set(self.world.residents)-{'player'}:
                self.population.recovery_required='Every autonomous resident must have its own learning brain. Saving and advancing are blocked; reload a complete checkpoint or restart.'
                raise RuntimeError(self.population.recovery_required)
            result=self.population.step(command)
        except Exception as error:
            self.manual_pause=self.paused=True
            self.controller_error=f'{type(error).__name__}: {error}'
            recovery=getattr(self.population,'recovery_required',None)
            if recovery and recovery not in self.controller_error:self.controller_error+=' — '+recovery
            self.pause_reason='A resident model needs attention; world paused'
            self._message(self.pause_reason+' — '+self.controller_error)
            return False
        self.pending_action=None
        self.controller_error=None
        if command:self._message(result['player'][1])
        self._samples.append(time.monotonic())
        return True

    def _save(self,filename):
        if getattr(self.population,'recovery_required',None):
            self._message(self.population.recovery_required)
            return False
        validate_roster(self.world,self.population.brains,self.population.metrics,
                        self.population.neighbors_scripted,require_all=self.all_residents)
        # World, optimizer, hidden states and pending transitions commit together.
        name='manual-population.pt' if filename=='world.json' else 'population.pt' if filename=='autosave.json' else Path(filename).stem+'-population.pt'
        self.population.save(self.runs/name)
        self.saved_at=datetime.now(timezone.utc).isoformat()
        if filename=='autosave.json':
            receipt=dict(saved_at=self.saved_at,tick=self.world.tick,checkpoint=name,
                         bytes=(self.runs/name).stat().st_size,residents=list(self.population.brains),
                         backend=getattr(self.population,'backend','one-step'),
                         scripted_residents=0 if not getattr(self.population,'neighbors_scripted',True) else len(self.world.residents)-len(self.population.brains)-1)
            temporary=self.runs/'checkpoint-status.tmp'
            temporary.write_text(json.dumps(receipt),encoding='utf-8')
            temporary.replace(self.runs/'checkpoint-status.json')

    def control(self,data):
        with self.lock:
            return self._control(data)

    def _control(self,data):
        if (getattr(self.population,'recovery_required',None)
                and data.get('command') in ('save','speed','resume','step')):
            return False,self.population.recovery_required
        if data.get('command')=='load':
            with self.lock:
                if self.public_demo:return False,'This shared demo is read-only.'
                path=self.runs/'manual-population.pt'
                if path.exists():loaded=load_population(path,all_residents=self.all_residents)
                elif self.migrate_world and (self.runs/'world.json').exists():
                    loaded=SequencePopulation.from_world(World.load(self.runs/'world.json'))
                else:return False,'Save this learning population first.'
                if self.all_residents:loaded.enable_all_residents()
                if not getattr(self.population,'recovery_required',None):
                    self.population.save(self.runs/'before-load-population.pt')
                self.population=loaded;self.world=loaded.world;self.pending_action=None
                self.controller_error=None
                self.manual_pause=self.paused=True
                self.speed=1;self.achieved_speed=0.;self._samples.clear()
                self.pause_reason='Loaded; paused by you'
                message=f'Restored world and {len(loaded.brains)} independent brains; paused.'
                self._message(message)
                return True,message
        ok,message=super().control(data)
        if data.get('command')=='step' and self.controller_error:
            self._message(self.pause_reason+' — '+self.controller_error)
            return False,self.last_message
        return ok,message

    def state_bytes(self,terrain=False,perspective='player'):
        with self.lock:
            result=json.loads(super().state_bytes(terrain,perspective))
            result['hasSave']=(self.runs/'manual-population.pt').exists() or (self.migrate_world and (self.runs/'world.json').exists())
            count=len(self.population.brains)
            laya=sum(getattr(b,'backend',None)=='laya-npu' for b in self.population.brains.values())
            result['controller']=(f'{count-laya} independent PPO learners + {laya} Laya model (frozen weights)'
                                  if laya else f'{count} independent neural learners · learning from their own experience')
            result['scriptedResidents']=0 if not getattr(self.population,'neighbors_scripted',True) else len(self.world.residents)-count-1
            result['controllerMode']='independent-learning'
            result['controllerError']=self.controller_error
            result['learning']={rid:{'updates':b.updates,'masked':b.masked,'decisions':b.decisions,**self.population.metrics[rid]} for rid,b in self.population.brains.items()}
            for rid,b in self.population.brains.items():
                if hasattr(b,'inspector'):result['learning'][rid].update(b.inspector())
            return json.dumps(result,separators=(',',':')).encode()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',default='runs/learning-population');p.add_argument('--ticks',type=int,default=256)
    p.add_argument('--serve',action='store_true');p.add_argument('--port',type=int,default=8789)
    p.add_argument('--paused',action='store_true',help='Start the interactive world paused, including with existing browser heartbeats')
    p.add_argument('--backend',choices=('one-step','recurrent-ppo'),default='recurrent-ppo')
    p.add_argument('--stage',choices=('near_food','depletion','hazards','farming','cooperation','scarcity'))
    args=p.parse_args();project=Path(__file__).resolve().parents[1];root=Path(args.directory).resolve()
    if root==project or not root.is_relative_to(project/'runs'):
        p.error('Use a separate directory inside this project runs folder')
    if args.ticks<0:p.error('ticks must be nonnegative')
    root.mkdir(parents=True,exist_ok=True)
    shutil.copytree(project/'web',root/'web',dirs_exist_ok=True)
    runtime=LearningRuntime(root,backend=args.backend,stage=args.stage)
    if args.paused:runtime.manual_pause=runtime.paused=True
    if not args.serve:
        start=time.perf_counter()
        for _ in range(args.ticks):runtime._tick()
        runtime.close()
        report={'tick':runtime.world.tick,'seconds':time.perf_counter()-start,'metrics':runtime.population.metrics,'updates':{rid:b.updates for rid,b in runtime.population.brains.items()}}
        (root/'latest-report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report));return
    server=WorldServer(('127.0.0.1',args.port),runtime)
    stop_file=root/'STOP'
    if stop_file.exists():stop_file.unlink()
    (root/'process.json').write_text(json.dumps({'pid':os.getpid(),'port':server.server_port,'script':str(Path(__file__).resolve())}))
    def watch():
        while not runtime.stop_event.wait(.2):
            if stop_file.exists():server.shutdown();return
    threading.Thread(target=watch,daemon=True).start()
    runtime.start();print(f'Experimental population: http://127.0.0.1:{server.server_port}/',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:runtime.close();server.server_close()


if __name__=='__main__':main()
