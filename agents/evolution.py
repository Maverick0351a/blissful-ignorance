"""Opt-in offspring generation; never changes a running resident or promotes automatically."""
import math
import random
import torch
from agents.replay_policy import Learner


def load_parent(path):
    """Read a locally generated tensor checkpoint without permitting pickled code."""
    data=torch.load(path,map_location='cpu',weights_only=True)
    model=Learner(data['seed'],data['curiosity']['coefficient'],hidden_size=data['config']['hidden_size'])
    model.network.load_state_dict(data['network']);model.target.load_state_dict(data['target'])
    model.optimizer.load_state_dict(data['optimizer'])
    model.replay.extend(data['replay']);model.replay_rng.setstate(data['replay_rng'])
    model.rng.setstate(data['action_rng']);model.updates=data['updates'];model.gamma=data['config']['gamma']
    model.curiosity.counts=dict(data['curiosity']['counts']);model.curiosity.last=data['curiosity']['last']
    return model


def _widen(source,destination):
    """Retain existing computation, leave new units available for future gradients."""
    with torch.no_grad():
        old=source[0].out_features
        destination[0].weight[:old].copy_(source[0].weight)
        destination[0].bias[:old].copy_(source[0].bias)
        destination[2].weight[:old].zero_()
        destination[2].weight[:old,:old].copy_(source[2].weight)
        destination[2].bias[:old].copy_(source[2].bias)
        destination[4].weight.zero_()
        destination[4].weight[:,:old].copy_(source[4].weight)
        destination[4].bias.copy_(source[4].bias)


def fork_candidate(parent,seed,hidden_size=None,learning_rate=None,curiosity=None,max_parameters=100000):
    """Copy learned knowledge into an independent, optionally larger offspring.

    The adjustable parameter budget controls this experiment's compute. It is
    not a fixed architecture limit. Optimizer and replay start fresh; weights
    and private novelty memories are inherited by value. No source rewriting.
    """
    width=parent.hidden_size if hidden_size is None else hidden_size
    if type(width) is not int or width<parent.hidden_size:
        raise ValueError('Growth must retain all existing hidden units')
    parameters=width*width+34*width+9
    if parameters>max_parameters:
        raise ValueError('Candidate exceeds this run parameter budget; increase it explicitly for a larger experiment')
    rate=parent.optimizer.param_groups[0]['lr'] if learning_rate is None else learning_rate
    coefficient=parent.curiosity.coefficient if curiosity is None else curiosity
    if not math.isfinite(rate) or rate<=0 or not math.isfinite(coefficient) or coefficient<0:
        raise ValueError('Invalid learning settings')
    child=Learner(seed,coefficient,hidden_size=width)
    _widen(parent.network,child.network);_widen(parent.target,child.target)
    for group in child.optimizer.param_groups:group['lr']=rate
    child.gamma=parent.gamma
    child.curiosity.counts=dict(parent.curiosity.counts);child.curiosity.last=parent.curiosity.last
    lineage={'parent_seed':parent.seed,'child_seed':seed,'parent_width':parent.hidden_size,
             'child_width':width,'parameters':parameters,'learning_rate':rate,'curiosity':coefficient,
             'optimizer':'fresh','replay':'fresh','promotion':'pending evaluation'}
    return child,lineage


def propose(parent,seed,max_parameters=100000):
    rng=random.Random(seed)
    width=parent.hidden_size+rng.choice((0,max(1,parent.hidden_size//4)))
    rate=parent.optimizer.param_groups[0]['lr']*rng.choice((.8,1.,1.2))
    coefficient=max(0.,parent.curiosity.coefficient+rng.choice((-.005,0.,.005)))
    return fork_candidate(parent,seed,width,rate,coefficient,max_parameters)


def promotion_gate(parent_rows,child_rows,min_seeds=3):
    """Conservative gate on matching fresh evaluations, not training reward.

    Rows are one actor-life each with seed/nutrition/zero_food_ticks/ticks/harvested.
    Parent and offspring need equal continuation-training budgets before this
    check; a caller must supply fresh evaluation seeds, never training seeds.
    """
    if len(parent_rows)!=len(child_rows) or not parent_rows:
        raise ValueError('Matched evaluations required')
    if [r['seed'] for r in parent_rows]!=[r['seed'] for r in child_rows]:
        raise ValueError('Evaluation seeds must match in order')
    if len(set(r['seed'] for r in child_rows))<min_seeds:
        return {'passed':False,'reason':'insufficient fresh evaluation seeds'}
    if any(p['ticks']!=c['ticks'] or c['ticks']<=0 for p,c in zip(parent_rows,child_rows)):
        raise ValueError('Evaluation horizons must match')
    mean=lambda rows,k:sum(r[k] for r in rows)/len(rows)
    nutrition_gain=mean(child_rows,'nutrition')>=max(mean(parent_rows,'nutrition')*1.05,mean(parent_rows,'nutrition')+1)
    no_worse=all(c['zero_food_ticks']<=p['zero_food_ticks'] for p,c in zip(parent_rows,child_rows))
    reliable=sum(r['zero_food_ticks']/r['ticks']<.01 and r['harvested']>0 for r in child_rows)/len(child_rows)>=.9
    return {'passed':nutrition_gain and no_worse and reliable,'nutrition_gain':nutrition_gain,
            'no_deprivation_regression':no_worse,'reliable_feeding':reliable}
