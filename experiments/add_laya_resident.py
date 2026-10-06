"""Add a distinct persistent Laya resident without replacing existing lives."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agents.laya_resident import LayaBrain
from agents.sequence_ppo import Population
from sim.checkpoint_lease import checkpoint_lease
from sim.world import Resident


def add_laya(population):
    if 'laya' in population.brains:
        if population.brains['laya'].backend != 'laya-npu':
            raise ValueError('The Laya identity already belongs to another model')
        return False
    if 'laya' in population.world.residents:
        raise ValueError('The Laya identity already has a body without its saved model')
    w = population.world
    traveler = w.residents['player']
    occupied = {(a.x,a.y) for a in w.residents.values()}
    # A new arrival needs an unoccupied walkable tile; no resource editing or route.
    sites = [(x,y) for y in range(len(w.terrain)) for x in range(len(w.terrain[y]))
             if w.walkable(x,y) and (x,y) not in occupied and w.key(x,y) not in w.resources
             and w.key(x,y) not in w.structures and w.key(x,y) not in w.caches]
    if not sites:raise ValueError('No clear arrival tile is available')
    x,y = min(sites, key=lambda p:(abs(p[0]-traveler.x)+abs(p[1]-traveler.y),p))
    w.residents['laya'] = Resident('laya', 'Laya', x, y, '#d9a1dd')
    population.brains['laya'] = LayaBrain(w.seed+91001)
    population.metrics['laya'] = population.new_metrics()
    population.neighbors_scripted = False
    w.record('Laya arrived with her own local model.', 'arrival', 'laya')
    return True


def main():
    runs=ROOT/'runs'
    with checkpoint_lease(runs):
        if (runs/'server-process.json').exists():raise RuntimeError('Stop the main server first')
        source=runs/'population.pt'
        original=hashlib.sha256(source.read_bytes()).hexdigest()
        p=Population.load(source,require_all=True)
        if not add_laya(p):
            print('Laya is already a persistent distinct resident.');return
        backup=runs/('before-laya-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.pt')
        shutil.copy2(source,backup)
        # Exercise loading the real existing NPU model before publishing the roster.
        p.brains['laya']._load_engine()
        if hashlib.sha256(source.read_bytes()).hexdigest()!=original:
            raise RuntimeError('The original population changed; not publishing')
        p.save(source)
        receipt=dict(saved_at=datetime.now(timezone.utc).isoformat(),tick=p.world.tick,
                     checkpoint='population.pt',bytes=source.stat().st_size,residents=list(p.brains),
                     backend='recurrent-ppo',scripted_residents=0)
        (runs/'checkpoint-status.json').write_text(json.dumps(receipt),encoding='utf-8')
        record=dict(**receipt,source_sha256=original,backup=str(backup),new_id='laya',
                    model='laya-npu',frozen_weights=True,body='Fresh arrival: 100 needs, empty pack')
        (runs/'laya-arrival.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
        print(json.dumps(record),flush=True)


if __name__=='__main__':main()
