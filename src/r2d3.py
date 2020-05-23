import tensorflow as tf
import rl
import rl.core
import keras
from keras.layers import Input, Flatten, Permute, TimeDistributed, LSTM, Dense, Concatenate, Reshape, Lambda
from keras.models import Model
from keras.models import model_from_json
from keras import backend as K
import numpy as np

import multiprocessing as mp
import math
import os
import pickle
import enum
import time
import traceback
import ctypes
import random

from .common import clipped_error_loss, rescaling, InputType, LstmType, DuelingNetwork
from .env_play import add_memory
from .memory import EpisodeMemory



# 複数のプロセスでGPUを使用する設定
# https://qiita.com/studio_haneya/items/4dfaf2fb2ac44818e7e0
# https://github.com/tensorflow/tensorflow/issues/11812
for device in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(device, True)


#---------------------------------------------------
# manager
#---------------------------------------------------
class R2D3():
    def __init__(self, 
        # model関係
        input_shape,
        input_type,
        nb_actions,  # アクション数(出力)
        memory,
        actors,
        optimizer,
        processor=None,
        metrics=[],
        image_model=None,  # imegeモデルを指定
        input_sequence=4,  # 入力フレーム数
        dense_units_num=512,          # Dense層のユニット数
        enable_dueling_network=True,  # dueling_network有効フラグ
        dueling_network_type=DuelingNetwork.AVERAGE,   # dueling_networkのアルゴリズム
        lstm_type=LstmType.NONE,  # LSTMのアルゴリズム
        lstm_units_num=512,       # LSTM層のユニット数
        lstm_ful_input_length=1,  # ステートフルLSTMの入力数
        batch_size=32,            # batch_size
        
        # learner 関係
        memory_warmup_size=100,  # 初期のメモリー確保用step数(学習しない)
        target_model_update=500,        #  target networkのupdate間隔
        enable_double_dqn=True,         # DDQN有効フラグ
        burnin_length=4,          # burn-in期間
        priority_exponent=0.9,    # シーケンス長priorityを計算する際のη

        # actor関係
        actor_model_sync_interval=500,  # learner から model を同期する間隔
        gamma=0.99,         # Q学習の割引率
        enable_rescaling=True,    # rescalingを有効にするか
        rescaling_epsilon=0.001,  # rescalingの定数
        reward_multisteps=3,  # multistep reward
        action_interval=1,    # アクションを実行する間隔

        demo_memory=None,
        demo_episode_dir="",
        demo_ratio_initial=1.0/256.0,
        demo_ratio_final=None,
        demo_ratio_steps=100_000,

        episode_memory=None,
        episode_ratio=1.0/256.0,
        episode_verbose=1,

        # その他
        enable_terminal_zero_reward=True,
        verbose=1,
    ):

        #--- check
        if lstm_type != LstmType.STATEFUL:
            burnin_length = 0
        
        assert memory.capacity > batch_size, "Memory capacity is small.(Larger than batch size)"
        assert memory_warmup_size > batch_size, "Warmup steps is few.(Larger than batch size)"

        if image_model is None:
            assert input_type == InputType.VALUES
        else:
            assert input_type == InputType.GRAY_2ch or input_type == InputType.GRAY_3ch or input_type == InputType.COLOR

            # 画像入力の制約
            # LSTMを使う場合: 画像は(w,h,ch)で入力できます。
            # LSTMを使わない場合：
            #   input_sequenceが1：全て使えます。
            #   input_sequenceが1以外：GRAY_2ch のみ使えます。
            if lstm_type == LstmType.NONE and input_sequence != 1:
                assert (input_type == InputType.GRAY_2ch), "input_iimage can use GRAY_2ch."

        #---
        self.kwargs = {
            "input_shape": input_shape,
            "input_type": input_type,
            "nb_actions": nb_actions,
            "memory": memory,
            "actors": actors,
            "optimizer": optimizer,
            "processor": processor,
            "metrics": metrics,
            "image_model": image_model,
            "input_sequence": input_sequence,
            "dense_units_num": dense_units_num,
            "enable_dueling_network": enable_dueling_network,
            "dueling_network_type": dueling_network_type,
            "lstm_type": lstm_type,
            "lstm_units_num": lstm_units_num,
            "lstm_ful_input_length": lstm_ful_input_length,
            "batch_size": batch_size,
            "memory_warmup_size": memory_warmup_size,
            "target_model_update": target_model_update,
            "enable_double_dqn": enable_double_dqn,
            "enable_rescaling": enable_rescaling,
            "rescaling_epsilon": rescaling_epsilon,
            "burnin_length": burnin_length,
            "priority_exponent": priority_exponent,
            "actor_model_sync_interval": actor_model_sync_interval,
            "gamma": gamma,
            "reward_multisteps": reward_multisteps,
            "action_interval": action_interval,
            "demo_memory": demo_memory,
            "demo_episode_dir": demo_episode_dir,
            "demo_ratio_initial": demo_ratio_initial,
            "demo_ratio_final": demo_ratio_final,
            "demo_ratio_steps": demo_ratio_steps,
            "episode_memory": episode_memory,
            "episode_ratio": episode_ratio,
            "episode_verbose": episode_verbose,
            "enable_terminal_zero_reward": enable_terminal_zero_reward,
            "verbose": verbose,
        }

        self.learner_ps = None
        self.actors_ps = []

    def __del__(self):
        if self.learner_ps is not None:
            self.learner_ps.terminate()
        for p in self.actors_ps:
            p.terminate()

    def train(self, 
            nb_trains,
            manager_allocate="/device:CPU:0",
            learner_allocate="/device:GPU:0",
            callbacks=[],
        ):

        # GPU確認
        # 参考: https://qiita.com/studio_haneya/items/4dfaf2fb2ac44818e7e0
        if len(tf.config.experimental.list_physical_devices('GPU')) > 0:
            self.enable_GPU = True
        else:
            self.enable_GPU = False

        #--- init
        self.kwargs["nb_trains"] = nb_trains
        self.kwargs["callbacks"] = R2D3CallbackList(callbacks)
        actor_num = len(self.kwargs["actors"])
        learner_allocate = learner_allocate
        verbose = self.kwargs["verbose"]

        if self.enable_GPU:
            self._train_allocate(manager_allocate, actor_num, learner_allocate, verbose)
        else:
            self._train(actor_num, learner_allocate, verbose)

    def _train_allocate(self, allocate, *args):
        with tf.device(allocate):
            self._train(*args)

    def _train(self, actor_num, learner_allocate, verbose):
    
        # 通信用変数
        self.learner_end_signal = mp.Value(ctypes.c_bool, False)
        self.is_learner_end = mp.Value(ctypes.c_bool, False)
        self.train_count = mp.Value(ctypes.c_int, 0)

        # 経験通信用
        exp_q = mp.Queue()
        
        weights_qs = []
        self.is_actor_ends = []
        for _ in range(actor_num):
            # model weights通信用
            weights_q = mp.Queue()
            weights_qs.append(weights_q)
            self.is_actor_ends.append(mp.Value(ctypes.c_bool, False))

        self.kwargs["callbacks"].on_r2d3_train_begin()
        t0 = time.time()
        try:

            # learner ps の実行
            learner_args = (
                self.kwargs,
                exp_q,
                weights_qs,
                self.learner_end_signal,
                self.is_learner_end,
                self.train_count,
            )
            if self.enable_GPU:
                learner_args = (learner_allocate,) + learner_args
                self.learner_ps = mp.Process(target=learner_run_allocate, args=learner_args)
            else:
                self.learner_ps = mp.Process(target=learner_run, args=learner_args)
            self.learner_ps.start()

            # actor ps の実行
            self.actors_ps = []
            for i in range(actor_num):
                # args
                actor_args = (
                    i,
                    self.kwargs,
                    exp_q,
                    weights_qs[i],
                    self.is_learner_end,
                    self.train_count,
                    self.is_actor_ends[i],
                )
                if self.enable_GPU:
                    actor = self.kwargs["actors"][i]
                    actor_args = (actor.allocate,) + actor_args
                    ps = mp.Process(target=actor_run_allocate, args=actor_args)
                else:
                    ps = mp.Process(target=actor_run, args=actor_args)
                self.actors_ps.append(ps)
                ps.start()

            # 終了を待つ
            while True:
                time.sleep(1)  # polling time

                # learner終了確認
                if self.is_learner_end.value:
                    break

                # actor終了確認
                f = True
                for is_actor_end in self.is_actor_ends:
                    if not is_actor_end.value:
                        f = False
                        break
                if f:
                    break
        
        except KeyboardInterrupt:
            pass
        except Exception:
            print(traceback.format_exc())
        if verbose > 0:
            print("done, took {:.3f} seconds".format(time.time() - t0))

        self.kwargs["callbacks"].on_r2d3_train_end()
        
        # learner に終了を投げる
        self.learner_end_signal.value = True

        # learner が終了するまで待つ
        t0 = time.time()
        while not self.is_learner_end.value:
            if time.time() - t0 < 360:  # timeout
                if verbose > 0:
                    print("learner end timeout.")
                    break
            time.sleep(1)

    def createTestAgent(self, test_actor, learner_model_path):
        return R2D3.createTestAgentStatic(self.kwargs, test_actor, learner_model_path)
        
    @staticmethod
    def createTestAgentStatic(manager_kwargs, test_actor, learner_model_path):
        test_actor = ActorRunner(-1, manager_kwargs, test_actor(), None, None, None, None)
        with open(learner_model_path, 'rb') as f:
            d = pickle.load(f)
        test_actor.model.set_weights(d["weights"])
        return test_actor


#---------------------------------------------------
# create model
#---------------------------------------------------
def build_compile_model(kwargs):
    input_shape = kwargs["input_shape"]
    input_type = kwargs["input_type"]
    image_model = kwargs["image_model"]
    batch_size = kwargs["batch_size"]
    input_sequence = kwargs["input_sequence"]
    lstm_type = kwargs["lstm_type"]
    lstm_units_num = kwargs["lstm_units_num"]
    enable_dueling_network = kwargs["enable_dueling_network"]
    dense_units_num = kwargs["dense_units_num"]
    nb_actions = kwargs["nb_actions"]
    dueling_network_type = kwargs["dueling_network_type"]
    optimizer = kwargs["optimizer"]
    metrics = kwargs["metrics"]


    if input_type == InputType.VALUES:
        if lstm_type != LstmType.STATEFUL:
            c = input_ = Input(shape=(input_sequence,) + input_shape)
        else:
            c = input_ = Input(batch_shape=(batch_size, input_sequence) + input_shape)
    elif input_type == InputType.GRAY_2ch:
        if lstm_type != LstmType.STATEFUL:
            c = input_ = Input(shape=(input_sequence,) + input_shape)
        else:
            c = input_ = Input(batch_shape=(batch_size, input_sequence) + input_shape)
    else:
        if lstm_type != LstmType.STATEFUL:
            c = input_ = Input(shape=input_shape)
        else:
            c = input_ = Input(batch_shape=(batch_size, input_sequence) + input_shape)

    if image_model is None:
        # input not image
        if lstm_type == LstmType.NONE:
            c = Flatten()(c)
        else:
            c = TimeDistributed(Flatten())(c)
    else:
        # input image
        if lstm_type == LstmType.NONE:
            enable_lstm = False
            if input_type == InputType.GRAY_2ch:
                # (input_seq, w, h) ->(w, h, input_seq)
                c = Permute((2, 3, 1))(c)
        elif lstm_type == LstmType.STATELESS or lstm_type == LstmType.STATEFUL:
            enable_lstm = True
            if input_type == InputType.GRAY_2ch:
                # (time steps, w, h) -> (time steps, w, h, ch)
                c = Reshape((input_sequence, ) + input_shape + (1,) )(c)
        else:
            raise ValueError('lstm_type is not undefined')
        c = image_model.create_image_model(c, enable_lstm)

    # lstm layer
    if lstm_type == LstmType.STATELESS:
        c = LSTM(lstm_units_num, name="lstm")(c)
    elif lstm_type == LstmType.STATEFUL:
        c = LSTM(lstm_units_num, stateful=True, name="lstm")(c)

    # dueling network
    if enable_dueling_network:
        # value
        v = Dense(dense_units_num, activation="relu")(c)
        v = Dense(1, name="v")(v)

        # advance
        adv = Dense(dense_units_num, activation='relu')(c)
        adv = Dense(nb_actions, name="adv")(adv)

        # 連結で結合
        c = Concatenate()([v,adv])
        if dueling_network_type == DuelingNetwork.AVERAGE:
            c = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:] - K.mean(a[:, 1:], axis=1, keepdims=True), output_shape=(nb_actions,))(c)
        elif dueling_network_type == DuelingNetwork.MAX:
            c = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:] - K.max(a[:, 1:], axis=1, keepdims=True), output_shape=(nb_actions,))(c)
        elif dueling_network_type == DuelingNetwork.NAIVE:
            c = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:], output_shape=(nb_actions,))(c)
        else:
            raise ValueError('dueling_network_type is not undefined')
    else:
        c = Dense(dense_units_num, activation="relu")(c)
        c = Dense(nb_actions, activation="linear", name="adv")(c)
    
    model = Model(input_, c)
    model.compile(loss=clipped_error_loss, optimizer=optimizer, metrics=metrics)
    
    return model


#---------------------------------------------------
# learner
#---------------------------------------------------
def learner_run_allocate(allocate, *args):
    with tf.device(allocate):
        learner_run(*args)

def learner_run(
        kwargs, 
        exp_q,
        weights_qs,
        learner_end_signal,
        is_learner_end,
        train_count,
    ):
    nb_trains = kwargs["nb_trains"]
    verbose = kwargs["verbose"]
    callbacks = kwargs["callbacks"]

    try:
        runner = LearnerRunner(kwargs, exp_q, weights_qs, train_count)
    
        callbacks.on_r2d3_learner_begin(runner)

        # learner はひたすら学習する
        if verbose > 0:
            print("Learner Start!")
        
        while True:
            callbacks.on_r2d3_learner_train_begin(runner)
            runner.train()
            callbacks.on_r2d3_learner_train_end(runner)

            # 終了判定
            if learner_end_signal.value:
                break

            # 終了判定
            if nb_trains > 0:
                if runner.train_count.value > nb_trains:
                    break
            
    except KeyboardInterrupt:
        pass
    except Exception:
        print(traceback.format_exc())

    try:
        if verbose > 0:
            print("Learning End. Train Count:{}".format(runner.train_count.value))

        callbacks.on_r2d3_learner_end(runner)
    except Exception:
        print(traceback.format_exc())

    is_learner_end.value = True

class LearnerRunner():
    def __init__(self,
            kwargs,
            exp_q,
            weights_qs,
            train_count,
        ):
        self.exp_q = exp_q
        self.weights_qs = weights_qs
        self.kwargs = kwargs
        self.actors_num = len(kwargs["actors"])

        self.input_shape = kwargs["input_shape"]
        self.enable_rescaling = kwargs["enable_rescaling"]
        self.memory = kwargs["memory"]
        self.memory_warmup_size = kwargs["memory_warmup_size"]
        self.gamma = kwargs["gamma"]
        self.batch_size = kwargs["batch_size"]
        self.enable_double_dqn = kwargs["enable_double_dqn"]
        self.target_model_update = kwargs["target_model_update"]
        self.input_sequence = kwargs["input_sequence"]
        self.lstm_type = kwargs["lstm_type"]
        self.burnin_length = kwargs["burnin_length"]
        self.priority_exponent = kwargs["priority_exponent"]
        self.actor_model_sync_interval = kwargs["actor_model_sync_interval"]
        self.reward_multisteps = kwargs["reward_multisteps"]
        self.lstm_ful_input_length = kwargs["lstm_ful_input_length"]
        self.demo_memory = kwargs["demo_memory"]
        self.enable_terminal_zero_reward = kwargs["enable_terminal_zero_reward"]

        if kwargs["episode_memory"] is None:
            self.episode_memory = None
            self.episode_ratio = 0
        else:
            self.episode_memory = EpisodeMemory(kwargs["episode_memory"], kwargs["episode_verbose"])
            self.episode_ratio = kwargs["episode_ratio"]

        if self.demo_memory is not None:
            add_memory(kwargs["demo_episode_dir"], self.demo_memory, self)
            assert len(self.demo_memory) > self.batch_size, \
                "Demo memory size is small."
            self.demo_ratio_initial = kwargs["demo_ratio_initial"]
            self.demo_ratio_final = kwargs["demo_ratio_final"]
            if self.demo_ratio_final is None:
                self.demo_ratio_final = kwargs["demo_ratio_initial"]
            else:
                self.demo_ratio_final = self.demo_ratio_final
            self.demo_ratio_step = \
                (self.demo_ratio_initial - self.demo_ratio_final) / kwargs["demo_ratio_steps"]
        else:
            self.demo_ratio_initial = 0
            self.demo_ratio_final = 0
            self.demo_ratio_step = 0

        # train_count
        self.train_count = train_count

        # model create
        self.model = build_compile_model(kwargs)
        self.target_model = build_compile_model(kwargs)

        if self.lstm_type == LstmType.STATEFUL:
            self.lstm = self.model.get_layer("lstm")
            self.target_lstm = self.target_model.get_layer("lstm")

        if self.episode_memory is not None:
            self.episode_exp = [ [] for _ in range(self.actors_num)]
            self.total_reward = [ 0 for _ in range(self.actors_num)]


    def train(self):
        _train_count = self.train_count.value
        
        # 一定毎に Actor に weights を送る
        if _train_count% self.actor_model_sync_interval == 0:
            weights = self.model.get_weights()
            for q in self.weights_qs:
                # 送る
                q.put(weights)
        
        # experience があれば RemoteMemory に追加
        for _ in range(self.exp_q.qsize()):
            exp = self.exp_q.get(timeout=1)

            # add memory
            self.memory.add(exp[0], exp[0][4])
            if self.episode_memory is not None:
                self.episode_exp[exp[1]].append(exp[0])
                if self.lstm_type == LstmType.STATEFUL:
                    self.total_reward[exp[1]] += exp[3]
                else:
                    self.total_reward[exp[1]] += exp[0][2]

            
            if exp[2]:  # terminal
                if self.enable_terminal_zero_reward and exp[2] != 0:
                    # 終了時に報酬が0以外なら0報酬の状態も追加
                    if self.lstm_type != LstmType.STATEFUL:
                        # STATEFUL はActor側で追加
                        exp2 = (exp[0][0], exp[0][1], 0, exp[0][3], exp[0][4])
                        self.memory.add(exp2, exp2[4])
                        if self.episode_memory is not None:
                            self.episode_exp[exp[1]].append(exp2)

                # episode_memory
                if self.episode_memory is not None:
                    self.episode_memory.add_episode(
                        self.episode_exp[exp[1]],
                        self.total_reward[exp[1]]
                    )
                    self.episode_exp[exp[1]] = []
                    self.total_reward[exp[1]] = 0
                    

        # RemoteMemory が一定数貯まるまで学習しない。
        if len(self.memory) <= self.memory_warmup_size:
            return

        # batch ratio
        batch_replay = 0
        batch_demo = 0
        batch_episode = 0

        ratio_demo = self.demo_ratio_initial - _train_count * self.demo_ratio_step
        if ratio_demo < self.demo_ratio_final:
            ratio_demo = self.demo_ratio_final
        if self.episode_memory is None or len(self.episode_memory) < self.batch_size:
            ratio_epi = 0
        else:
            ratio_epi = self.episode_ratio
        for _ in range(self.batch_size):
            r = random.random()
            if r < ratio_demo:
                batch_demo += 1
                continue
            r -= ratio_demo
            if r < ratio_epi:
                batch_episode += 1
            else:
                batch_replay += 1
        
        # memory から優先順位に基づき状態を取得
        indexes = []
        batchs = []
        weights = []
        memory_types = []
        if batch_replay > 0:
            (i, b, w) = self.memory.sample(batch_replay, _train_count)
            indexes.extend(i)
            batchs.extend(b)
            weights.extend(w)
            memory_types.extend([0 for _ in range(batch_replay)])
        if batch_demo > 0:
            (i, b, w) = self.demo_memory.sample(batch_demo, _train_count)
            indexes.extend(i)
            batchs.extend(b)
            weights.extend(w)
            memory_types.extend([1 for _ in range(batch_demo)])
        if batch_episode > 0:
            (i, b, w) = self.episode_memory.sample(batch_episode, _train_count)
            indexes.extend(i)
            batchs.extend(b)
            weights.extend(w)
            memory_types.extend([2 for _ in range(batch_episode)])
        
        # 学習(長いので関数化)
        if self.lstm_type == LstmType.STATEFUL:
            self.train_model_ful(indexes, batchs, weights, memory_types)
        else:
            self.train_model(indexes, batchs, weights, memory_types)
        self.train_count.value += 1  # 書き込みは一人なのでlockは不要

        # target networkの更新
        if self.train_count.value % self.target_model_update == 0:
            self.target_model.set_weights(self.model.get_weights())
    
    # ノーマルの学習
    def train_model(self, indexes, batchs, weights, memory_types):
        state0_batch = []
        action_batch = []
        reward_batch = []
        state1_batch = []
        for batch in batchs:
            state0_batch.append(batch[0])
            action_batch.append(batch[1])
            reward_batch.append(batch[2])
            state1_batch.append(batch[3])
        state0_batch = np.asarray(state0_batch)
        state1_batch = np.asarray(state1_batch)

        # 更新用に現在のQネットワークを出力(Q network)
        state0_qvals = self.model.predict(state0_batch, self.batch_size)

        if self.enable_double_dqn:
            # TargetNetworkとQNetworkのQ値を出す
            state1_qvals_model = self.model.predict(state1_batch, self.batch_size)
            state1_qvals_target = self.target_model.predict(state1_batch, self.batch_size)
        else:
            # 次の状態のQ値を取得(target_network)
            state1_qvals_target = self.target_model.predict(state1_batch, self.batch_size)

        for i in range(self.batch_size):
            if self.enable_double_dqn:
                action = state1_qvals_model[i].argmax()  # modelからアクションを出す
                maxq = state1_qvals_target[i][action]  # Q値はtarget_modelを使って出す
            else:
                maxq = state1_qvals_target[i].max()

            # priority計算
            q0 = state0_qvals[i][action_batch[i]]
            td_error = reward_batch[i] + (self.gamma ** self.reward_multisteps) * maxq - q0
            priority = abs(td_error)

            # Q値の更新
            state0_qvals[i][action_batch[i]] += td_error * weights[i]

            # priorityを更新
            if memory_types[i] == 0:
                self.memory.update(indexes[i], batchs[i], priority)
            elif memory_types[i] == 1:
                self.demo_memory.update(indexes[i], batchs[i], priority)
            elif memory_types[i] == 2:
                self.episode_memory.update(indexes[i], batchs[i], priority)
            else:
                assert False

        # 学習
        self.model.train_on_batch(state0_batch, state0_qvals)


    # ステートフルLSTMの学習
    def train_model_ful(self, indexes, batchs, weights, memory_types):

        hidden_s0 = []
        hidden_s1 = []
        for batch in batchs:
            # batchサイズ分あるけどすべて同じなので0番目を取得
            hidden_s0.append(batch[3][0][0])
            hidden_s1.append(batch[3][1][0])
        hidden_states = [np.asarray(hidden_s0), np.asarray(hidden_s1)]

         # init hidden_state
        self.lstm.reset_states(hidden_states)
        self.target_lstm.reset_states(hidden_states)

        # predict
        hidden_states_arr = []
        if self.burnin_length == 0:
            hidden_states_arr.append(hidden_states)
        state_batch_arr = []
        model_qvals_arr = []
        target_qvals_arr = []
        prioritys = [ [] for _ in range(self.batch_size)]
        for seq_i in range(self.burnin_length + self.reward_multisteps + self.lstm_ful_input_length):

            # state
            state_batch = [ batch[0][seq_i] for batch in batchs ]
            state_batch = np.asarray(state_batch)
            
            # hidden_state更新およびQ値取得
            model_qvals = self.model.predict(state_batch, self.batch_size)
            target_qvals = self.target_model.predict(state_batch, self.batch_size)

            # burnin-1
            if seq_i < self.burnin_length-1:
                continue
            hidden_states_arr.append([K.get_value(self.lstm.states[0]), K.get_value(self.lstm.states[1])])

            # burnin
            if seq_i < self.burnin_length:
                continue

            state_batch_arr.append(state_batch)
            model_qvals_arr.append(model_qvals)
            target_qvals_arr.append(target_qvals)

        # train
        for seq_i in range(self.lstm_ful_input_length):

            # state0 の Qval (multistep前)
            state0_qvals = model_qvals_arr[seq_i]
            
            # batch
            for batch_i in range(self.batch_size):

                # maxq
                if self.enable_double_dqn:
                    action = model_qvals_arr[seq_i+self.reward_multisteps][batch_i].argmax()  # modelからアクションを出す
                    maxq = target_qvals_arr[seq_i+self.reward_multisteps][batch_i][action]  # Q値はtarget_modelを使って出す
                else:
                    maxq = target_qvals_arr[seq_i+self.reward_multisteps][batch_i].max()

                # priority
                batch_action = batchs[batch_i][1][seq_i]
                q0 = state0_qvals[batch_i][batch_action]
                reward = batchs[batch_i][2][seq_i]
                td_error = reward + (self.gamma ** self.reward_multisteps) * maxq - q0
                priority = abs(td_error)
                prioritys[batch_i].append(priority)

                # Q値の更新
                state0_qvals[batch_i][batch_action] += td_error * weights[batch_i]

            # train
            self.lstm.reset_states(hidden_states_arr[seq_i])
            self.model.train_on_batch(state_batch_arr[seq_i], state0_qvals)
            
        # priority update
        for i, batch in enumerate(batchs):
            priority = self.priority_exponent * np.max(prioritys[i]) + \
                (1-self.priority_exponent) * np.average(prioritys[i])

            # priorityを更新
            if memory_types[i] == 0:
                self.memory.update(indexes[i], batch, priority)
            elif memory_types[i] == 1:
                self.demo_memory.update(indexes[i], batch, priority)
            elif memory_types[i] == 2:
                self.episode_memory.update(indexes[i], batch, priority)
            else:
                assert False

        
    def save_weights(self, filepath, overwrite=False, save_memory=False):
        if overwrite or not os.path.isfile(filepath):
            d = {
                "weights": self.model.get_weights(),
                "train": self.train_count.value,
            }
            with open(filepath, 'wb') as f:
                pickle.dump(d, f)

            # memory
            if save_memory:
                d = {}
                d["replay"] = self.memory.get_memorys()
                if self.episode_memory is not None:
                    d["episode"] = self.episode_memory.get_memorys()
                with open(filepath + ".mem", 'wb') as f:
                    pickle.dump(d, f)
        
    def load_weights(self, filepath, load_memory=False):
        with open(filepath, 'rb') as f:
            d = pickle.load(f)
        self.model.set_weights(d["weights"])
        self.target_model.set_weights(d["weights"])
        self.train_count.value = d["train"]

        # memory
        if load_memory:
            filepath = filepath + ".mem"
            if os.path.isfile(filepath):
                with open(filepath, 'rb') as f:
                    d = pickle.load(f)
                self.memory.set_memorys(d["replay"])
                if "episode" in d and self.episode_memory is not None:
                    self.episode_memory.set_memorys(d["episode"])


#---------------------------------------------------
# actor
#---------------------------------------------------
class ActorStop(rl.callbacks.Callback):
    def __init__(self, is_learner_end):
        self.is_learner_end = is_learner_end

    def on_step_end(self, episode, logs={}):
        if self.is_learner_end.value:
            raise KeyboardInterrupt()

class Actor():
    allocate = "/device:CPU:0"

    def getPolicy(self, actor_index, actor_num):
        raise NotImplementedError()

    def fit(self, index, agent):
        raise NotImplementedError()


def actor_run_allocate(allocate, *args):
    with tf.device(allocate):
        actor_run(*args)

def actor_run(
        actor_index,
        kwargs, 
        exp_q,
        weights_q,
        is_learner_end,
        train_count,
        is_actor_end,
    ):
    verbose = kwargs["verbose"]
    callbacks = kwargs["callbacks"]

    actor = kwargs["actors"][actor_index]()

    runner = ActorRunner(
        actor_index,
        kwargs,
        actor,
        exp_q,
        weights_q,
        is_learner_end,
        train_count,
    )

    try:
        callbacks.on_r2d3_actor_begin(actor_index, runner)

        # run
        if verbose > 0:
            print("Actor{} Start!".format(actor_index))
        actor.fit(actor_index, runner)
        
    except KeyboardInterrupt:
        pass
    except Exception:
        print(traceback.format_exc())
        
    try:
        if verbose > 0:
            print("Actor{} End!".format(actor_index))
        callbacks.on_r2d3_actor_end(actor_index, runner)
    except Exception:
        print(traceback.format_exc())

    is_actor_end.value = True



class ActorRunner(rl.core.Agent):
    def __init__(self, 
            actor_index,
            kwargs,
            actor,
            exp_q,
            weights_q,
            is_learner_end,
            train_count,
        ):
        super(ActorRunner, self).__init__(kwargs["processor"])
        self.is_learner_end = is_learner_end
        self.train_count = train_count
        self.callbacks = kwargs.get("callbacks", [])

        self.actor_index = actor_index
        self.actor = actor
        self.exp_q = exp_q
        self.weights_q = weights_q
        self.actors_num = len(kwargs["actors"])

        self.enable_rescaling = kwargs["enable_rescaling"]
        self.rescaling_epsilon = kwargs["rescaling_epsilon"]
        self.action_policy = actor.getPolicy(actor_index, self.actors_num)
        self.nb_actions = kwargs["nb_actions"]
        self.input_shape = kwargs["input_shape"]
        self.input_sequence = kwargs["input_sequence"]
        self.gamma = kwargs["gamma"]
        self.reward_multisteps = kwargs["reward_multisteps"]
        self.action_interval = kwargs["action_interval"]
        self.burnin_length = kwargs["burnin_length"]
        self.lstm_type = kwargs["lstm_type"]
        self.enable_dueling_network = kwargs["enable_dueling_network"]
        self.priority_exponent = kwargs["priority_exponent"]
        self.lstm_ful_input_length = kwargs["lstm_ful_input_length"]
        self.batch_size = kwargs["batch_size"]
        self.enable_terminal_zero_reward = kwargs["enable_terminal_zero_reward"]
        self.verbose = kwargs["verbose"]

        self.enable_episode_memory = kwargs["episode_memory"] is not None

        # create model
        self.model = build_compile_model(kwargs)
        if self.lstm_type == LstmType.STATEFUL:
            self.lstm = self.model.get_layer("lstm")
        model_json = self.model.to_json()
        self.action_policy.compile(model_json)
        self.compiled = True  # super


    def reset_states(self):  # override
        self.repeated_action = 0
        self.recent_terminal = False

        if self.lstm_type == LstmType.STATEFUL:
            multi_len = self.reward_multisteps + self.lstm_ful_input_length - 1
            self.recent_actions = [ 0 for _ in range(multi_len + 1)]
            self.recent_rewards = [ 0 for _ in range(multi_len)]
            self.recent_rewards_multistep = [ 0 for _ in range(self.lstm_ful_input_length)]
            tmp = self.burnin_length + self.input_sequence + multi_len
            self.recent_observations = [
                np.zeros(self.input_shape) for _ in range(tmp)
            ]
            tmp = self.burnin_length + multi_len + 1
            self.recent_observations_wrap = [
                [np.zeros(self.input_shape) for _ in range(self.input_sequence)] for _ in range(tmp)
            ]

            # hidden_state: [(batch_size, lstm_units_num), (batch_size, lstm_units_num)]
            tmp = self.burnin_length + multi_len + 1+1
            self.model.reset_states()
            self.recent_hidden_states = [
                [K.get_value(self.lstm.states[0]), K.get_value(self.lstm.states[1])] for _ in range(tmp)
            ]
            
        else:
            self.recent_actions = [ 0 for _ in range(self.reward_multisteps+1)]
            self.recent_rewards = [ 0 for _ in range(self.reward_multisteps)]
            self.recent_rewards_multistep = 0
            self.recent_observations = [
                np.zeros(self.input_shape) for _ in range(self.input_sequence + self.reward_multisteps)
            ]
    
    def compile(self, optimizer, metrics=[]):  # override
        self.compiled = True  # super

    def save_weights(self, filepath, overwrite=False):  # override
        if overwrite or not os.path.isfile(filepath):
            filepath = filepath.format(index=self.actor_index)
            d = self.action_policy.get_weights()
            with open(filepath, 'wb') as f:
                pickle.dump(d, f)

    def load_weights(self, filepath):  # override
        filepath = filepath.format(index=self.actor_index)
        with open(filepath, 'rb') as f:
            d = pickle.load(f)
        self.action_policy.set_weights(d)

    def forward(self, observation):  # override
        # observation
        self.recent_observations.pop(0)
        self.recent_observations.append(observation)

        if self.lstm_type == LstmType.STATEFUL:
            self.recent_observations_wrap.pop(0)
            self.recent_observations_wrap.append(self.recent_observations[-self.input_sequence:])
            
            # tmp
            self._state0 = self.recent_observations_wrap[-self.burnin_length -1]

        else:
            # tmp
            self._state0 = self.recent_observations[:self.input_sequence]

        # tmp
        self._qvals = None
        self._state1 = self.recent_observations[-self.input_sequence:]
        self._state1_np = np.asarray(self._state1)
        self._state0_np = np.asarray(self._state0)

        if self.training:

            # experienceを送る
            if self.lstm_type == LstmType.STATEFUL:
                
                #--- priorityを計算
                # 初回しか使わないので計算量のかかるburn-inは省略
                # (直前のhidden_statesなのでmodelによる誤差もほぼないため)

                prioritys = []
                for i in range(self.lstm_ful_input_length):

                    state0 = self._state0_np
                    state1 = self._state1_np
                    hidden_states0 = self.recent_hidden_states[self.burnin_length + i]
                    hidden_states1 = self.recent_hidden_states[self.burnin_length + i + self.reward_multisteps]
                    action = self.recent_actions[i]
                    reward = self.recent_rewards_multistep[i]

                    # batchサイズ分増やす
                    state0_batch = np.full((self.batch_size,)+state0.shape, state0)
                    state1_batch = np.full((self.batch_size,)+state1.shape, state1)

                    # 現在のQネットワークを出力
                    self.lstm.reset_states(hidden_states0)
                    state0_qvals = self.model.predict(state0_batch, self.batch_size)[0]
                    self.lstm.reset_states(hidden_states1)
                    state1_qvals = self.model.predict(state1_batch, self.batch_size)[0]

                    maxq = np.max(state1_qvals)
                    td_error = reward + (self.gamma ** self.reward_multisteps) * maxq
                    priority = abs(td_error - state0_qvals[action])
                    prioritys.append(priority)
                
                # 今回使用したsamplingのpriorityを更新
                priority = self.priority_exponent * np.max(prioritys) + (1-self.priority_exponent) * np.average(prioritys)
                
                # RemoteMemory に送信
                if self.recent_terminal and self.recent_rewards[-1] == 0 and self.enable_terminal_zero_reward:
                    # 報酬が0以外は最後を追加して送る。
                    self.exp_q.put(((
                            self.recent_observations_wrap[:],
                            self.recent_actions[0:self.lstm_ful_input_length],
                            self.recent_rewards_multistep[:],
                            self.recent_hidden_states[0],
                            priority,
                        ),
                        self.actor_index, 
                        False,
                        self.recent_rewards[-1],
                    ))

                    e0 = self.recent_observations_wrap[1:]
                    e0.append(e0[-1])
                    e1 = self.recent_actions[1:self.lstm_ful_input_length]
                    e1.append(e1[-1])
                    e2 = self.recent_rewards_multistep[1:]
                    e2.append(0)
                    exp2 = (e0, e1, e2, self.recent_hidden_states[1], priority)
                    self.exp_q.put((
                        exp2,
                        self.actor_index,
                        True,
                        0,
                    ))

                else:
                    # そのまま送る
                    self.exp_q.put(((
                            self.recent_observations_wrap[:],
                            self.recent_actions[0:self.lstm_ful_input_length],
                            self.recent_rewards_multistep[:],
                            self.recent_hidden_states[0],
                            priority,
                        ),
                        self.actor_index, 
                        self.recent_terminal,
                        self.recent_rewards[-1],
                    ))

            else:

                state0 = self._state0_np[np.newaxis,:]
                state1 = self._state1_np[np.newaxis,:]
                action = self.recent_actions[0]
                reward = self.recent_rewards_multistep

                #--- priority の計算
                state0_qvals = self.model.predict(state0, 1)[0]
                state1_qvals = self.model.predict(state1, 1)[0]
                maxq = np.max(state1_qvals)
                td_error = reward + (self.gamma ** self.reward_multisteps) * maxq - state0_qvals[action]
                priority = abs(td_error)

                # RemoteMemory に送信
                self.exp_q.put(((
                        self._state0, 
                        action, 
                        reward, 
                        self._state1,
                        priority,
                    ),
                    self.actor_index, 
                    self.recent_terminal
                ))


        # 状態の更新
        if self.lstm_type == LstmType.STATEFUL:
            self.lstm.reset_states(self.recent_hidden_states[-1])

            # hidden_state を更新しつつQ値も取得
            state = self._state1_np
            state = np.full((self.batch_size,)+state.shape, state)  # batchサイズ分増やす
            self._qvals = self.model.predict(state, batch_size=self.batch_size)[0]
            
            hidden_state = [K.get_value(self.lstm.states[0]), K.get_value(self.lstm.states[1])]
            self.recent_hidden_states.pop(0)
            self.recent_hidden_states.append(hidden_state)

        if self.recent_terminal:
            return 0  # 終了時はactionを出す必要がない


        # フレームスキップ(action_interval毎に行動を選択する)
        action = self.repeated_action
        if self.step % self.action_interval == 0:
            
            # 行動を決定
            if self.training:
                # training中かつNoisyNetが使ってない場合は action policyに従う
                action = self.action_policy.select_action(self)
            else:
                # テスト中またはNoisyNet中の場合
                action = np.argmax(self.get_qvals())
            
            # リピート用
            self.repeated_action = action

        # アクション保存
        self.recent_actions.pop(0)
        self.recent_actions.append(action)

        return action
        

    def get_qvals(self):
        if self.lstm_type == LstmType.STATEFUL:
            return self._qvals
        else:
            if self._qvals is None:
                state = self._state1_np[np.newaxis,:]
                self._qvals = self.model.predict(state, batch_size=1)[0]
            return self._qvals
    
    def get_state(self):
        return self._state1_np
    
    def get_prev_state(self):
        if self.lstm_type == LstmType.STATEFUL:
            observation = self._state0_np
            action = self.recent_actions[-self.reward_multisteps-1]
            reward = self.recent_rewards_multistep[-self.reward_multisteps]
        else:
            observation = self._state0_np
            action = self.recent_actions[0]
            reward = self.recent_rewards_multistep
        return (observation, action, reward)

    def backward(self, reward, terminal):  # override
        # terminal は env が終了状態ならTrue
        if not self.training:
            return []

        # 報酬の保存
        self.recent_rewards.pop(0)
        self.recent_rewards.append(reward)

        # multi step learning の計算
        _tmp = 0
        for i in range(-self.reward_multisteps, 0):
            r = self.recent_rewards[i]
            _tmp += r * (self.gamma ** i)
        
        # rescaling
        if self.enable_rescaling:
            _tmp = rescaling(_tmp)
        
        if self.lstm_type == LstmType.STATEFUL:
            self.recent_rewards_multistep.pop(0)
            self.recent_rewards_multistep.append(_tmp)
        else:
            self.recent_rewards_multistep = _tmp

        # weightが届いていればmodelを更新
        if not self.weights_q.empty():
            weights = self.weights_q.get(timeout=1)
            # 空にする(念のため)
            while not self.weights_q.empty():
                self.weights_q.get(timeout=1)
            self.model.set_weights(weights)

        self.recent_terminal = terminal

        return []

    @property
    def layers(self):  # override
        return self.model.layers[:]


    def fit(self, env, nb_steps=99_999_999_999, callbacks=[], **kwargs):  # override

        if self.actor_index == -1:
            # test_actor
            super().fit(nb_steps, callbacks, **kwargs)
            return

        callbacks.extend(self.callbacks.callbacks)

        # stop
        callbacks.append(ActorStop(self.is_learner_end))

        # keras-rlでの学習
        super().fit(env, nb_steps=nb_steps, callbacks=callbacks, **kwargs)




class R2D3Callback(rl.callbacks.Callback):
    def __init__(self):
        pass

    def on_r2d3_train_begin(self):
        pass

    def on_r2d3_train_end(self):
        pass

    def on_r2d3_learner_begin(self, learner):
        pass
    
    def on_r2d3_learner_end(self, learner):
        pass

    def on_r2d3_learner_train_begin(self, learner):
        pass

    def on_r2d3_learner_train_end(self, learner):
        pass

    def on_r2d3_actor_begin(self, actor_index, runner):
        pass

    def on_r2d3_actor_end(self, actor_index, runner):
        pass

class R2D3CallbackList(R2D3Callback):
    def __init__(self, callbacks):
        self.callbacks = callbacks

    def on_r2d3_train_begin(self):
        for callback in self.callbacks:
            callback.on_r2d3_train_begin()

    def on_r2d3_train_end(self):
        for callback in self.callbacks:
            callback.on_r2d3_train_end()

    def on_r2d3_learner_begin(self, learner):
        for callback in self.callbacks:
            callback.on_r2d3_learner_begin(learner)
    
    def on_r2d3_learner_end(self, learner):
        for callback in self.callbacks:
            callback.on_r2d3_learner_end(learner)

    def on_r2d3_learner_train_begin(self, learner):
        for callback in self.callbacks:
            callback.on_r2d3_learner_train_begin(learner)

    def on_r2d3_learner_train_end(self, learner):
        for callback in self.callbacks:
            callback.on_r2d3_learner_train_end(learner)

    def on_r2d3_actor_begin(self, actor_index, runner):
        for callback in self.callbacks:
            callback.on_r2d3_actor_begin(actor_index, runner)

    def on_r2d3_actor_end(self, actor_index, runner):
        for callback in self.callbacks:
            callback.on_r2d3_actor_end(actor_index, runner)


