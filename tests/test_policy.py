import unittest

import numpy as np
from keras.models import Model
from keras.layers import Dense, Input

import random

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from src.policy import *
from src.common import LstmType


class StubAgent():
    def __init__(self):
        self.nb_actions = 5
        self.lstm_type = LstmType.NONE

        self.model = self.build_compile_model()

    def build_compile_model(self):
        #--- model
        c = input_ = Input(shape=(3,))
        #c = Flatten()(c)
        c = Dense(16, activation="relu")(c)
        c = Dense(self.nb_actions, activation="linear")(c)
        model = Model(input_, c)
        model.compile(optimizer='sgd', loss='mse')
        return model

    def get_qvals(self):
        return [0.5, 0.4, 0.3, 0.2, 0.1]

    def get_state(self):
        observation = [random.random(), random.random(), random.random()]
        return np.asarray(observation)
    
    def get_prev_state(self):
        observation = [random.random(), random.random(), random.random()]
        action = 3
        reward = random.random()
        return (np.asarray(observation), action, reward)


class Test(unittest.TestCase):

    def test_policies(self):
        test_patterns = [
            EpsilonGreedy(epsilon=0.4),
            AnnealingEpsilonGreedy(0.4, 0.01, 1000),
            EpsilonGreedyActor(0, 10),
            SoftmaxPolicy(),
            UCB1(),
            UCB1_Tuned(),
            UCBv(),
            #KL_UCB(),
            ThompsonSamplingBeta(),
            ThompsonSamplingGaussian(),
        ]

        for policy in test_patterns:
            with self.subTest(policy.__class__.__name__):
                self._testPolicy(policy)


    def _testPolicy(self, policy):
        agent = StubAgent()
        check_actions = [i for i in range(agent.nb_actions)]

        #--- compile
        model_json = agent.model.to_json()
        policy.compile(model_json)

        counter = {}
        for i in range(1000):
            action = policy.select_action(agent)
            self.assertIn(action, check_actions)

            #--- 定期的にget/setを挟む
            if i%1000 == 0:
                d = policy.get_weights()
                policy.set_weights(d)
            
            if action not in counter:
                counter[action] = 0
            counter[action] += 1

        #print(counter)

        # 0～nb_action
        counter_keys = list(counter.keys())
        counter_keys.sort()
        self.assertEqual(counter_keys, check_actions)

        #--- 比率チェック
        if policy.__class__ == EpsilonGreedy or policy.__class__ == AnnealingEpsilonGreedy:
            # ほとんど4
            n1 = 0
            n2 = 0
            for k, v in counter.items():
                if k == 0:
                    n2 += v
                else:
                    n1 += v
            self.assertLess(n1, n2)
        elif policy.__class__ == SoftmaxPolicy:
            # 順番に減る
            for i in range(agent.nb_actions-1):
                self.assertLess(counter[i+1], counter[i])


if __name__ == "__main__":
    unittest.main()
