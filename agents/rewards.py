"""Local bodily feedback and private scene novelty for learning experiments."""
import hashlib
import json
import math


def hunger_reward(nutrition, discomfort_integral):
    """One full meal earns 1; 480 ticks at maximum hunger cost 1.

    The integral is sum of the agent's own 0..100 discomfort over ticks.
    Unconscious ticks are included, so fainting cannot escape the cost.
    """
    if not all(math.isfinite(v) and v >= 0 for v in (nutrition, discomfort_integral)):
        raise ValueError('Expected finite nonnegative bodily measurements')
    return nutrition / 25 - discomfort_integral / 48000


def scene_key(observation):
    """Only local visible scene structure; exclude identity, body, clock and amounts."""
    scene=[]
    for t in observation.get('tiles', ()):
        if t.get('terrain', -1) < 0:
            continue
        r=t.get('resource', {}); s=t.get('structure', {})
        scene.append((t['dx'],t['dy'],t['terrain'],r.get('kind'),s.get('kind'),bool(s.get('open'))))
    return hashlib.sha256(json.dumps(sorted(scene),separators=(',',':')).encode()).hexdigest()


class Curiosity:
    """Per-agent novelty memory. A revisited scene gives a smaller bonus.

    Repeated identical views give zero; no global map or shared counts are used.
    This is engineered intrinsic motivation, not proof of subjective curiosity.
    """
    def __init__(self, coefficient=.02):
        self.coefficient=coefficient
        self.counts={}
        self.last=None

    def reward(self, observation, eligible=True):
        if not eligible:
            return 0.
        key=scene_key(observation)
        if key==self.last:
            return 0.
        self.last=key
        self.counts[key]=self.counts.get(key,0)+1
        return self.coefficient/math.sqrt(self.counts[key])

    def snapshot(self):
        return {'coefficient':self.coefficient,'counts':dict(self.counts),'last':self.last}
