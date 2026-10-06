import unittest
from sim.world import World
from sim.social import TONES, AUDITORY_WINDOW, AUDITORY_CAPACITY


class ToneSequenceTests(unittest.TestCase):
    def world(self):
        w=World(22);w.resources.clear();w.structures.clear()
        w.terrain=[[0]*64 for _ in range(64)]
        w.residents={k:w.residents[k] for k in ('player','r0','r1')}
        for rid,pos in zip(w.residents,((32,32),(33,32),(48,48))):
            w.residents[rid].x,w.residents[rid].y=pos
        return w

    def step(self,w,command=None):
        return w.step({rid:command if rid=='player' and command else {'verb':'wait'} for rid in w.residents})

    def send(self,w,symbol):
        self.assertTrue(self.step(w,{'verb':'tone','tone':symbol})['player'][0])
        self.step(w);self.step(w)

    def test_ten_choices_and_nonoverlap(self):
        w=self.world()
        for symbol in TONES:
            self.assertTrue(self.step(w,{'verb':'tone','tone':symbol})['player'][0])
            self.assertEqual([s['kind'] for s in w.hearing(w.residents['r0'])],['tone_'+symbol])
            self.assertFalse(self.step(w,{'verb':'tone','tone':symbol})['player'][0])
            self.assertFalse(w.hearing(w.residents['r0']))
            self.assertFalse(self.step(w,{'verb':'tone','tone':symbol})['player'][0])
        self.assertFalse(w.apply_action('player',{'verb':'tone','tone':'10'})[0])

    def test_order_repetition_and_encoder(self):
        import torch
        from agents.network import encode_observation
        worlds=[]
        for sequence in (('2','7','2'),('7','2','2')):
            w=self.world()
            for symbol in sequence:self.send(w,symbol)
            observation=w.observe('r0');events=observation['auditory_memory']
            self.assertEqual([e['tone'] for e in events],list(sequence))
            self.assertEqual([e['age'] for e in events],[9,6,3])
            self.assertEqual(set(events[0]),{'tone','bearing','strength','age'})
            self.assertFalse(w.observe('r1')['auditory_memory'])
            self.assertFalse(w.observe('player')['auditory_memory'])
            worlds.append(encode_observation(observation)[1])
        self.assertFalse(torch.equal(*worlds))

    def test_save_expiry_no_retroactive_hearing(self):
        w=self.world();self.send(w,'8')
        clone=World.from_dict(w.to_dict())
        self.assertEqual(clone.observe('r0')['auditory_memory'],w.observe('r0')['auditory_memory'])
        w.residents['r1'].x,w.residents['r1'].y=32,33
        self.step(w)
        self.assertFalse(w.observe('r1')['auditory_memory'])
        w.tick=AUDITORY_WINDOW
        self.assertFalse(w.observe('r0')['auditory_memory'])
        self.step(w);self.assertFalse(w.auditory_memories['r0'])
        old=clone.to_dict();old.pop('auditory_memories')
        self.assertFalse(World.from_dict(old).observe('r0')['auditory_memory'])
        # Old tone names are migrated; external aliases remain compatible.
        old['sounds']=[{'x':32,'y':32,'actor':'player','kind':'tone_high','radius':6,'tick':old['tick']}]
        self.assertEqual(World.from_dict(old).sounds[0]['kind'],'tone_2')

    def test_bounded_copy_and_no_unconscious_acquisition(self):
        w=self.world();w.residents['r0'].unconscious=True;w.residents['r0'].health=1
        self.send(w,'0');self.assertFalse(w.observe('r0')['auditory_memory'])
        w.residents['r0'].unconscious=False
        self.send(w,'1')
        w.auditory_memories['r0']*=50
        events=w.observe('r0')['auditory_memory']
        self.assertEqual(len(events),AUDITORY_CAPACITY)
        events[0]['tone']='wrong'
        self.assertEqual(w.observe('r0')['auditory_memory'][0]['tone'],'1')


if __name__=='__main__':unittest.main()
