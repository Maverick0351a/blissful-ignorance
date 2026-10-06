"""Helper tests run before locked World integration; integrated tests follow patch."""
import unittest
from sim.world import World
from sim.social import attention, expire_attention, tap


class SocialTests(unittest.TestCase):
    def setUp(self):
        self.w = World(11)
        self.w.residents = {k: self.w.residents[k] for k in ('player', 'r0', 'r1')}
        self.w.resources.clear(); self.w.structures.clear()
        self.w.terrain = [[0]*64 for _ in range(64)]
        for rid, pos in zip(self.w.residents, ((32,32), (32,33), (33,32))):
            self.w.residents[rid].x, self.w.residents[rid].y = pos

    def test_gentle_recipient_only_and_expiry(self):
        before = [(a.health,a.pain,a.food) for a in self.w.residents.values()]
        self.assertTrue(tap(self.w, 'player', {})[0])
        self.assertEqual(self.w.residents['r0'].facing, 'north')
        self.assertEqual(attention(self.w, 'r0'), [{'tick':0,'bearing':'north','kind':'tap'}])
        self.assertFalse(attention(self.w, 'player')); self.assertFalse(attention(self.w, 'r1'))
        self.assertEqual(before, [(a.health,a.pain,a.food) for a in self.w.residents.values()])
        self.w.tick = 15; self.assertTrue(attention(self.w,'r0'))
        self.w.tick = 16; expire_attention(self.w); self.assertEqual(self.w.attention_events,{})

    def test_target_selection_and_copy(self):
        self.assertTrue(tap(self.w,'player',{'target':'r1'})[0])
        event = attention(self.w,'r1')[0]; event['bearing']='wrong'
        self.assertEqual(attention(self.w,'r1')[0]['bearing'],'west')
        self.w.tick = 8
        self.assertFalse(tap(self.w,'player',{'target':'missing'})[0])
        self.assertTrue(tap(self.w,'player',{})[0])
        self.assertTrue(attention(self.w,'r0'))

    def test_reach_occlusion_and_unconscious(self):
        self.w.residents['r0'].unconscious=True
        self.assertFalse(tap(self.w,'player',{'target':'r0'})[0])
        self.assertTrue(self.w.residents['r0'].unconscious)
        self.w.residents['r1'].x = 34
        self.assertFalse(tap(self.w,'player',{})[0])
        self.w.residents['r0'].unconscious=False
        self.w.structures['32,33']={'kind':'wall'}
        self.assertFalse(tap(self.w,'player',{'target':'r0'})[0])
        self.w.residents['player'].unconscious=True
        self.assertFalse(tap(self.w,'player',{})[0])

    def test_cooldown_and_bounded_events(self):
        self.assertTrue(tap(self.w,'player',{})[0])
        self.w.tick=7; self.assertFalse(tap(self.w,'player',{})[0])
        self.w.tick=8; self.assertTrue(tap(self.w,'player',{})[0])
        # Stored input may be large; observation keeps only the bounded recent tail.
        self.w.attention_events['r0'] *= 9
        self.assertLessEqual(len(attention(self.w,'r0')),4)

    def test_integrated_save_and_order(self):
        from unittest.mock import patch
        for order in [('player','r0','r1'), ('r1','r0','player')]:
            self.setUp()
            with patch.object(self.w.rng, 'shuffle', side_effect=lambda ids: ids.__setitem__(slice(None),order)):
                self.w.step({'player':{'verb':'tap'},'r0':{'verb':'rest'},'r1':{'verb':'rest'}})
            self.assertEqual(self.w.residents['r0'].facing,'north')
            self.assertTrue(self.w.observe('r0')['attention'])
            self.assertFalse(self.w.observe('r1')['attention'])
            clone=World.from_dict(self.w.to_dict())
            self.assertEqual(clone.observe('r0')['attention'],self.w.observe('r0')['attention'])
            self.assertEqual(clone.residents['player'].last_tap_tick,0)
        data=self.w.to_dict();data.pop('attention_events')
        for a in data['residents']:
            a.pop('last_tap_tick');a.pop('last_tone_tick')
        self.assertEqual(World.from_dict(data).attention_events,{})

    def test_tones_range_occlusion_decay_privacy_and_save(self):
        w=self.w
        self.assertTrue(w.apply_action('player',{'verb':'tone','tone':'9'})[0])
        first=w.observe('r0')['hearing'][0]
        self.assertEqual(first['kind'],'tone_9')
        self.assertEqual(first['bearing'],'north')
        self.assertEqual(set(first),{'kind','bearing','strength','age'})
        self.assertFalse(w.observe('player')['hearing'])
        self.assertFalse(w.apply_action('player',{'verb':'tone','tone':'9'})[0])
        w.residents['r1'].x=40
        self.assertFalse(w.observe('r1')['hearing'])
        w.residents['r0'].y=35
        clear=w.observe('r0')['hearing'][0]['strength']
        w.structures['32,34']={'kind':'wall'}
        self.assertLess(w.observe('r0')['hearing'][0]['strength'],clear)
        clone=World.from_dict(w.to_dict())
        self.assertEqual(clone.observe('r0')['hearing'],w.observe('r0')['hearing'])
        self.assertFalse(clone.apply_action('player',{'verb':'tone'})[0])
        w.tick=8
        self.assertFalse(w.observe('r0')['hearing'])
        self.assertTrue(w.apply_action('r0',{'verb':'tone','tone':'0'})[0])
        self.assertFalse(w.apply_action('player',{'verb':'tone','tone':[]})[0])
        w.residents['player'].unconscious=True
        self.assertFalse(w.apply_action('player',{'verb':'tone'})[0])

    def test_network_senses_and_actions(self):
        import torch
        from agents.network import encode_observation,ACTIONS,FEATURES
        before=encode_observation(self.w.observe('r0'))[1]
        self.w.apply_action('player',{'verb':'tap'})
        self.w.apply_action('player',{'verb':'tone','tone':'0'})
        after=encode_observation(self.w.observe('r0'))[1]
        self.assertEqual(after.shape,(1,FEATURES))
        self.assertFalse(torch.equal(before,after))
        self.assertIn({'verb':'tap'},ACTIONS)
        self.assertIn({'verb':'tone','tone':'0'},ACTIONS)


if __name__ == '__main__': unittest.main()
