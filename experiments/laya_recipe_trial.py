"""Explicit-recipe frozen Laya diagnostic; separate from mobile independent learners."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments import farming_trial as base

SEED=79021
DECISIONS=160


def choice(engine, adapter, brief, recent, mask, rotation):
    legal=[name for i,name in enumerate(base.VERBS) if mask[i]]
    offset=rotation%len(legal)
    legal=legal[offset:]+legal[:offset]
    state={'senses':brief,'recent':recent[-3:]}
    question={'action':{'type':'choice','instructions':
        'Keep fed over 160 turns. Start:2 food. Eating gives up to25 fullness; each turn costs0.96 fullness. make_seeds uses1 food to give2 seeds. plant uses1 seed on empty plot. A crop becomes3 food after4 turns; gather takes1 food. Neighbors may take ripe food. Stages0 empty,1-3 growing,4 ripe. Preserve a way to grow more food before eating the last food.',
        'criteria':{name:None for name in legal}}}
    seq,_=adapter.build_sequence(engine.tok,state,adapter.to_internal(question['action']),4096,engine.cfg.get('head_max_len',192))
    if len(seq)>adapter.L:raise ValueError(f'Input truncation: {len(seq)} > {adapter.L}')
    answer=engine.system_one(state,question)['answers']['action']['choice']
    return base.VERBS.index(answer)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--laya-adapter',required=True)
    args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    plan={'seed':SEED,'decisions':DECISIONS,'conditions':['solo','scripted-competitor'],
          'hypothesis':'Explicit recipe Laya converts, plants, harvests and eats more than2 meals in both long-life conditions.',
          'limits':'One seed diagnostic; fixed plot, frozen pretrained weights. Prior80-turn prompt is not matched horizon, so no causal effect-size claim about teaching.',
          'hashes':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ['experiments/laya_recipe_trial.py','experiments/farming_trial.py','sim/world.py']}}
    (out/'preregistration.json').write_text(json.dumps(plan,indent=2))
    os.environ['HF_HUB_OFFLINE']=os.environ['TRANSFORMERS_OFFLINE']='1'
    spec=importlib.util.spec_from_file_location('local_laya',args.laya_adapter)
    adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter)
    engine=adapter.LayaLite('NPU')
    base.DECISIONS=DECISIONS
    base.laya_choice=choice
    rows=[]
    for competition in (False,True):
        for kind in ('scripted','laya'):
            row,trace=base.episode(kind,SEED,competition,base.Learner(SEED),engine=engine,adapter=adapter)
            row['kind']=kind;rows.append(row)
            (out/f'{kind}-{competition}.json').write_text(json.dumps(trace))
            (out/'results.json').write_text(json.dumps(rows,indent=2))
            print(json.dumps(row),flush=True)
    (out/'complete.json').write_text(json.dumps({'complete':True,'rows':len(rows)}))


if __name__=='__main__':main()
