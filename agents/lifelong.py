"""Experimental independent online actor-critic learners; no pretrained behavior."""
from pathlib import Path
import random
import torch
from agents.affordances import action_mask
from agents.physiology import food_relief
from agents.preservation import decision_transaction, require_consistent, validate_roster
from agents.network import ResidentNetwork, encode_observation, ACTIONS
from sim.world import World

torch.set_num_threads(1)


class Brain:
    def __init__(self,seed,masked=True):
        with torch.random.fork_rng():
            torch.manual_seed(seed);self.model=ResidentNetwork(hidden_size=32)
        self.optimizer=torch.optim.Adam(self.model.parameters(),lr=.0003)
        self.rng=random.Random(seed);self.hidden=self.model.initial_state()
        self.masked=masked;self.training=True
        self.pending=None;self.reward=0.;self.updates=0;self.decisions=0

    def decide(self,observation):
        patch,features=encode_observation(observation)
        mask=torch.tensor([action_mask(observation,ACTIONS) if self.masked else [True]*len(ACTIONS)])
        if self.pending is not None and self.training:
            old_patch,old_features,old_hidden,chosen=self.pending[:4]
            old_mask=self.pending[4] if len(self.pending)>4 else torch.ones((1,len(ACTIONS)),dtype=torch.bool)
            logits,value,_=self.model(old_patch,old_features,old_hidden)
            with torch.no_grad():_,future,_=self.model(patch,features,self.hidden)
            advantage=self.reward+.99*future-value
            distribution=torch.distributions.Categorical(logits=logits.masked_fill(~old_mask,-1e9))
            loss=-(distribution.log_prob(torch.tensor([chosen]))*advantage.detach()).mean()+.5*advantage.square().mean()-.005*distribution.entropy().mean()
            self.optimizer.zero_grad();loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.)
            self.optimizer.step();self.updates+=1
        with torch.no_grad():
            logits,_,hidden=self.model(patch,features,self.hidden)
            probabilities=logits.masked_fill(~mask,-1e9).softmax(-1)[0].tolist()
        # Mask only locally known physical failures. No strategy is preferred.
        chosen=self.rng.choices(range(len(ACTIONS)),weights=probabilities)[0]
        self.pending=(patch,features,self.hidden,chosen,mask) if self.training else None
        self.hidden=tuple(t.detach() for t in hidden)
        self.reward=0.;self.decisions+=1
        return dict(ACTIONS[chosen])

    def state(self):
        return dict(masked=self.masked,training=self.training,model=self.model.state_dict(),optimizer=self.optimizer.state_dict(),rng=self.rng.getstate(),
                    hidden=self.hidden,pending=self.pending,reward=self.reward,updates=self.updates,decisions=self.decisions)

    def restore(self,data):
        self.model.load_state_dict(data['model']);self.optimizer.load_state_dict(data['optimizer'])
        self.rng.setstate(data['rng'])
        self.masked=data.get('masked',False);self.training=data.get('training',True)
        for key in ('hidden','pending','reward','updates','decisions'):setattr(self,key,data[key])


class Population:
    def __init__(self,seed=5729,masked=True,*,resident_ids=('r0','r1')):
        self.world=World(seed);self.world.ecology_enabled=True
        self.brains={rid:Brain(seed+i,masked) for i,rid in enumerate(resident_ids)}
        self.neighbors_scripted=True
        for rid in self.brains:
            self.world.residents[rid].food=50
            self.world.residents[rid].inventory.update(food=2,wood=3)
        self.metrics={rid:{'invalid':0,'nutrition':0.,'zero_food_ticks':0,'unconscious_ticks':0} for rid in self.brains}

    def enable_all_residents(self):
        require_consistent(self)
        validate_roster(self.world,self.brains,self.metrics,self.neighbors_scripted,require_all=True)
        masked=next(iter(self.brains.values())).masked
        for i,rid in enumerate(rid for rid in self.world.residents if rid!='player'):
            if rid not in self.brains:
                self.brains[rid]=Brain(self.world.seed+i,masked)
                self.metrics[rid]={'invalid':0,'nutrition':0.,'zero_food_ticks':0,'unconscious_ticks':0}
        self.neighbors_scripted=False

    def step(self,player_command=None):
        require_consistent(self)
        if self.world.tick%4==0:
            with decision_transaction(self):
                commands={rid:brain.decide(self.world.observe(rid)) for rid,brain in self.brains.items()}
        else:
            commands={rid:{'verb':'wait'} for rid in self.brains}
        try:
            return self._advance(commands,player_command)
        except BaseException:
            self.recovery_required='World transition failed. Saving and advancing are blocked; reload a complete checkpoint or restart.'
            raise

    def _advance(self,commands,player_command):
        if not self.neighbors_scripted:
            commands.update({rid:{'verb':'rest'} for rid in self.world.residents if rid not in self.brains})
        if player_command:commands['player']=player_command
        before={rid:(self.world.residents[rid].food,self.world.residents[rid].health,self.world.residents[rid].unconscious) for rid in self.brains}
        results=self.world.step(commands,scripted=self.neighbors_scripted)
        for rid,brain in self.brains.items():
            actor=self.world.residents[rid];food,health,was_unconscious=before[rid]
            relief=food_relief(food,actor.food)
            brain.reward+=relief/25-min(1.,max(0.,(60-actor.food)/60))*.005+min(0.,actor.health-health)/100
            m=self.metrics[rid];m['nutrition']+=relief;m['zero_food_ticks']+=actor.food<=0;m['unconscious_ticks']+=actor.unconscious
            if self.world.tick%4==1:
                m['invalid']+=not results[rid][0]
                m.setdefault('conscious_decisions',0);m.setdefault('conscious_invalid',0)
                if not was_unconscious:
                    m['conscious_decisions']+=1;m['conscious_invalid']+=not results[rid][0]
        return results

    def save(self,path):
        require_consistent(self)
        validate_roster(self.world,self.brains,self.metrics,self.neighbors_scripted)
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        temporary=path.with_suffix('.tmp')
        torch.save(dict(schema=1,world=self.world.to_dict(),neighbors_scripted=self.neighbors_scripted,
                        brains={rid:b.state() for rid,b in self.brains.items()},metrics=self.metrics),temporary)
        temporary.replace(path)

    @classmethod
    def load(cls,path,*,require_all=False):
        data=torch.load(path,map_location='cpu',weights_only=True)
        if data['schema']!=1:raise ValueError('Unsupported experimental checkpoint')
        world=World.from_dict(data['world'])
        validate_roster(world,data['brains'],data['metrics'],data.get('neighbors_scripted',True),require_all=require_all)
        result=cls(world.seed,resident_ids=tuple(data['brains']));result.world=world
        for rid,brain in result.brains.items():brain.restore(data['brains'][rid])
        result.metrics=data['metrics'];result.neighbors_scripted=data.get('neighbors_scripted',True);return result
