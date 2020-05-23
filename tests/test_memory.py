import unittest
import random
import time

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from src.memory import *

class Test(unittest.TestCase):

    #@unittest.skip("")
    def test_memorys(self):
        test_patterns = [
            ReplayMemory(10),
            PERGreedyMemory(10),
            PERProportionalMemory(
                capacity=10,
                alpha=0.8,
                beta_initial=1,
                beta_steps=10,
                enable_is=True
            ),
            PERRankBaseMemory(
                capacity=10,
                alpha=0.8,
                beta_initial=1,
                beta_steps=10,
                enable_is=True
            ),
        ]
        for memory in test_patterns:
            with self.subTest(memory.__class__.__name__):
                self._testMemory(memory)

    def _testMemory(self, memory):
        capacity = memory.capacity
        self.assertEqual(capacity, 10)

        # add
        for i in range(100):
            memory.add( (i,i,i,i), 0)
        self.assertEqual(len(memory), capacity)

        # 中身を1～10にする
        for i in range(capacity):
            i += 1
            memory.add((i,i,i,i) , i)
        assert len(memory) == capacity

        #--- 複数回やって比率をだす
        counter = {}
        for i in range(10000):
            (indexes, batchs, weights) = memory.sample(5, 1)
            self.assertEqual(len(indexes), 5)
            self.assertEqual(len(batchs),  5)
            self.assertEqual(len(weights), 5)
            
            # 重複がないこと
            li_uniq = list(set(batchs))
            self.assertEqual(len(li_uniq), 5, li_uniq)
            
            # batch count
            for batch in batchs:
                if batch[0] not in counter:
                    counter[batch[0]] = 0
                counter[batch[0]] += 1

            # update priority
            for i in range(5):
                memory.update(indexes[i], batchs[i], batchs[i][3])
            assert len(memory) == capacity

            # save/load
            d = memory.get_memorys()
            memory.set_memorys(d)
            d2 = memory.get_memorys()
            self.assertListEqual(d, d2)
            

        # debug
        #for k, v in sorted(counter.items(), key=lambda x: x[0]):
        #    print(str(k) + ": " + str(v))
        

        #--- 要素の確認
        counter_keys = list(counter.keys())
        counter_keys.sort()
        if memory.__class__ == PERGreedyMemory:
            # 0PERGreedyMemory は 6～10固定
            self.assertEqual(counter_keys, [i+1 for i in range(5, 10)])
        else:
            # 1～11まであること
            self.assertEqual(counter_keys, [i+1 for i in range(capacity)])
        
        #--- 比率チェック
        if memory.__class__ == ReplayMemory:
            pass
        elif memory.__class__ == PERGreedyMemory:
            # 全て同じ
            for i in range(6, 10):
                self.assertEqual(counter[i], counter[i+1])
        else:
            # priorityが高いほど数が増えている
            for i in range(capacity-1):
                i += 1
                self.assertLess(counter[i], counter[i+1])
    
    #@unittest.skip("")
    def test_speed(self):
        capacity = 100_000_000
        test_patterns = [
            ReplayMemory(capacity),
            PERGreedyMemory(capacity),
            PERProportionalMemory(
                capacity=capacity,
                alpha=0.8,
                beta_initial=1,
                beta_steps=10,
                enable_is=True
            ),
            PERRankBaseMemory(
                capacity=capacity,
                alpha=0.8,
                beta_initial=1,
                beta_steps=10,
                enable_is=True
            ),
        ]
        for memory in test_patterns:
            with self.subTest(memory.__class__.__name__):
                self._testSpeed(memory)


    def _testSpeed(self, memory):
        t0 = time.time()
        batch_size = 3

        # warmup
        uniqid = 0
        for _ in range(1000 + batch_size):
            r = random.random()
            memory.add( (uniqid,uniqid,uniqid,uniqid), r)
            uniqid += 1
        
        for _ in range(20_000):

            # add
            r = random.random()
            memory.add( (uniqid,uniqid,uniqid,uniqid), r)
            uniqid += 1

            # sample
            (indexes, batchs, weights) = memory.sample(batch_size, uniqid)
            self.assertEqual(len(indexes), batch_size)
            self.assertEqual(len(batchs),  batch_size)
            self.assertEqual(len(weights), batch_size)
            
            # 重複がないこと
            li_uniq = list(set(batchs))
            self.assertEqual(len(li_uniq), batch_size)
            
            # update priority
            for i in range(batch_size):
                r = random.random()
                memory.update(indexes[i], batchs[i], r)

        print("{}: {}s".format(memory.__class__.__name__, time.time()-t0))


    #@unittest.skip("")
    def test_is(self):
        test_patterns = [
            PERProportionalMemory(
                capacity=10,
                alpha=1.0,
                beta_initial=1,
                beta_steps=10,
                enable_is=True
            ),
            PERRankBaseMemory(
                capacity=10,
                alpha=1.0,
                beta_initial=1,
                beta_steps=10,
                enable_is=True
            ),
        ]
        for memory in test_patterns:
            with self.subTest(memory.__class__.__name__):
                self._test_is(memory)

    def _test_is(self, memory):
        for i in range(3):
            memory.add((i,i,i,i+1) , i+1)  # 最後をpriorityに

        (indexes, batchs, weights) = memory.sample(3, 1)
        maxw = 2.0
        for i in range(3):
            if batchs[i][0] == 0:
                priority = 1/6
                w = (3*priority) ** (-1)
                w /= maxw
                self.assertEqual(round(weights[i]-w,7), 0)
            elif batchs[i][0] == 1:
                priority = (3-1)/6
                w = (3*priority) ** (-1)
                w /= maxw
                self.assertEqual(round(weights[i]-w,7), 0)
            elif batchs[i][0] == 2:
                priority = (6-3)/6
                w = (3*priority) ** (-1)
                w /= maxw
                self.assertEqual(round(weights[i]-w,7), 0)


if __name__ == "__main__":
    unittest.main()
