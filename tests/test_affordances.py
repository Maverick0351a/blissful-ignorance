import copy
import unittest
from agents.affordances import action_mask
from agents.network import ACTIONS
from agents.lifelong import Population


class AffordanceTests(unittest.TestCase):
    def setUp(self):
        self.p=Population(81);self.w=self.p.world;self.a=self.w.residents['r0']
        self.w.resources.clear();self.w.structures.clear();self.w.terrain=[[0]*64 for _ in range(64)]
        self.a.x=self.a.y=20
        for key in self.a.inventory:self.a.inventory[key]=0

    def allows(self,command):return action_mask(self.w.observe('r0'),ACTIONS)[ACTIONS.index(command)]

    def test_physical_filters_keep_harmful_options(self):
        self.assertFalse(self.allows({'verb':'eat'}));self.assertFalse(self.allows({'verb':'gather'}))
        self.assertFalse(self.allows({'verb':'plant'}));self.assertTrue(self.allows({'verb':'rest'}))
        self.a.inventory['amber_fruit']=1;self.a.food=50
        self.assertTrue(self.allows({'verb':'eat','item':'amber_fruit'}))
        self.w.resources['21,20']={'kind':'thorns','amount':1}
        self.assertTrue(self.allows({'verb':'gather'}))
        self.assertTrue(self.allows({'verb':'move','direction':'east'}))
        self.w.structures['20,19']={'kind':'wall'}
        self.assertFalse(self.allows({'verb':'move','direction':'north'}))
        self.a.unconscious=True
        self.assertEqual(sum(action_mask(self.w.observe('r0'),ACTIONS)),1)

    def test_no_privileged_inputs_and_basket_capacity(self):
        observation=self.w.observe('r0');altered=copy.deepcopy(observation)
        altered.update(world_resources={'secret':999},tick=12345,player=True)
        self.assertEqual(action_mask(observation,ACTIONS),action_mask(altered,ACTIONS))
        self.w.caches['20,20']={'food':1};self.a.carried_cache={'food':7}
        self.assertTrue(self.allows({'verb':'take_food'}))
        self.a.inventory['wood']=1
        self.assertFalse(self.allows({'verb':'take_food'}))

    def test_chosen_actions_obey_masks_and_old_checkpoint_defaults(self):
        brain=self.p.brains['r0'];observation=self.w.observe('r0')
        for _ in range(12):
            action=brain.decide(observation)
            self.assertTrue(action_mask(observation,ACTIONS)[ACTIONS.index(action)])
        state=brain.state();state.pop('masked');state.pop('training');state['pending']=state['pending'][:4]
        brain.restore(state);self.assertFalse(brain.masked)
        brain.masked=True;brain.decide(observation)


if __name__=='__main__':unittest.main()
