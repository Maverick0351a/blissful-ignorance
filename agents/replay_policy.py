"""Independent CPU Double-DQN policies for a sealed farming experiment."""
from collections import deque
import random
import torch
from torch import nn
from agents.rewards import Curiosity

torch.set_num_threads(1)

def encode(state):
    """Encode the existing local, partially observed 13-field sensor state."""
    values = [float(state[i])/4 for i in (0,1,2,4,6,7,9,10,11)]
    values += [float(state[8]), float(state[12])]
    for index in (3,5):
        values += [float(state[index] == code) for code in (-1,0,1,2,3,4)]
    return torch.tensor(values, dtype=torch.float32)

class FrozenCuriosity(Curiosity):
    def reward(self, observation, eligible=True):
        return 0.

class Learner:
    def __init__(self, seed, coefficient=.02, hidden_size=64):
        self.seed=seed
        self.hidden_size=hidden_size
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self.network=nn.Sequential(nn.Linear(23,hidden_size),nn.ReLU(),nn.Linear(hidden_size,hidden_size),nn.ReLU(),nn.Linear(hidden_size,9))
        self.target=nn.Sequential(nn.Linear(23,hidden_size),nn.ReLU(),nn.Linear(hidden_size,hidden_size),nn.ReLU(),nn.Linear(hidden_size,9))
        self.target.load_state_dict(self.network.state_dict())
        self.optimizer=torch.optim.Adam(self.network.parameters(),lr=.001)
        self.rng=random.Random(seed)
        self.replay_rng=random.Random(seed ^ 0xDB12)
        self.replay=deque(maxlen=12000)
        self.trace=deque()
        self.q={}
        self.curiosity=Curiosity(coefficient)
        self.updates=0
        self.gamma=.995
    def choose(self,state,mask,training):
        allowed=[i for i,valid in enumerate(mask) if valid]
        if training: self.q[state]=True
        if training and self.rng.random()<.25:
            return self.rng.choice(allowed)
        with torch.no_grad():
            scores=self.network(encode(state))
        return max(allowed,key=lambda i:float(scores[i]))
    def update(self,state,action,reward,next_state,mask,terminal):
        self.trace.append((state,action,reward,next_state,tuple(mask),terminal))
        if len(self.trace)>=3:
            self._emit()
        if terminal:
            while self.trace: self._emit()
        if len(self.replay)>=64: self.train_batch()
    def _emit(self):
        sequence=list(self.trace)[:3]
        reward=sum(self.gamma**i*t[2] for i,t in enumerate(sequence))
        first,last=sequence[0],sequence[-1]
        self.replay.append((first[0],first[1],reward,last[3],last[4],last[5],len(sequence)))
        self.trace.popleft()
    def targets(self, rewards, states, masks, terminals, lengths):
        with torch.no_grad():
            scores=self.network(states).masked_fill(~masks,float('-inf'))
            actions=scores.argmax(1)
            future=self.target(states).gather(1,actions[:,None]).squeeze(1)
            return rewards + self.gamma**lengths * future * (~terminals)
    def train_batch(self):
        batch=self.replay_rng.sample(list(self.replay),64)
        states=torch.stack([encode(t[0]) for t in batch])
        next_states=torch.stack([encode(t[3]) for t in batch])
        actions=torch.tensor([t[1] for t in batch])
        targets=self.targets(torch.tensor([t[2] for t in batch]),next_states,
            torch.tensor([t[4] for t in batch]),torch.tensor([t[5] for t in batch]),torch.tensor([t[6] for t in batch]))
        predictions=self.network(states).gather(1,actions[:,None]).squeeze(1)
        loss=nn.functional.smooth_l1_loss(predictions,targets)
        self.optimizer.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(),5.)
        self.optimizer.step(); self.updates+=1
        if self.updates%100==0: self.target.load_state_dict(self.network.state_dict())
    def freeze_curiosity(self):
        previous=self.curiosity
        self.curiosity=FrozenCuriosity(previous.coefficient)
        self.curiosity.counts=dict(previous.counts)
        self.curiosity.last=previous.last
    def save(self,path):
        torch.save(dict(seed=self.seed,network=self.network.state_dict(),target=self.target.state_dict(),
            optimizer=self.optimizer.state_dict(),replay=list(self.replay),replay_rng=self.replay_rng.getstate(),
            action_rng=self.rng.getstate(),updates=self.updates,curiosity=self.curiosity.snapshot(),
            config=dict(gamma=self.gamma,replay_maxlen=self.replay.maxlen,hidden_size=self.hidden_size,dimensions=[23,self.hidden_size,self.hidden_size,9])),path)

class RandomLearner(Learner):
    def choose(self,state,mask,training):
        return self.rng.choice([i for i,v in enumerate(mask) if v])
