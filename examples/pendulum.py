import gym
from keras.optimizers import Adam

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from src.r2d3 import Actor
from src.processor import PendulumProcessorForDQN
from src.policy import EpsilonGreedy, AnnealingEpsilonGreedy
from src.image_model import DQNImageModel
from src.memory import PERRankBaseMemory, PERProportionalMemory
from src.common import InputType, LstmType, DuelingNetwork, seed_everything, LoggerType

from Lib import run_gym_rainbow, run_gym_r2d3, run_play, run_replay


seed_everything(42)
ENV_NAME = "Pendulum-v0"
episode_save_dir = "tmp_{}.".format(ENV_NAME)


def create_parameter():
    
    env = gym.make(ENV_NAME)
    
    # ゲーム情報
    print("action_space      : " + str(env.action_space))
    print("observation_space : " + str(env.observation_space))
    print("reward_range      : " + str(env.reward_range))

    image = False
    if image:
        processor = PendulumProcessorForDQN(enable_image=True, reward_clip=None)
        input_shape = processor.image_shape
        input_type = InputType.GRAY_2ch
        image_model = DQNImageModel()
        enable_rescaling = True
    else:
        processor = PendulumProcessorForDQN(enable_image=False, reward_clip=None)
        input_shape = env.observation_space.shape
        input_type = InputType.VALUES
        image_model = None
        enable_rescaling = True

    kwargs = {
        "input_shape": input_shape, 
        "input_type": input_type,
        "nb_actions": processor.nb_actions, 
        "optimizer": Adam(lr=0.001),
        "metrics": [],

        "image_model": image_model,
        "input_sequence": 4,         # 入力フレーム数
        "dense_units_num": 32,       # dense層のユニット数
        "enable_dueling_network": True,
        "dueling_network_type": DuelingNetwork.AVERAGE,  # dueling networkで使うアルゴリズム
        "lstm_type": LstmType.STATELESS,           # 使用するLSTMアルゴリズム
        "lstm_units_num": 32,             # LSTMのユニット数
        "lstm_ful_input_length": 2,       # ステートフルLSTMの入力数

        "memory_warmup_size": 200,    # 初期のメモリー確保用step数(学習しない)
        "target_model_update": 1000,  # target networkのupdate間隔
        "action_interval": 1,       # アクションを実行する間隔
        "batch_size": 8,     # batch_size
        "gamma": 0.99,        # Q学習の割引率
        "enable_double_dqn": True,
        "enable_rescaling": enable_rescaling,   # rescalingを有効にするか
        "rescaling_epsilon": 0.001,  # rescalingの定数
        "priority_exponent": 0.9,   # priority優先度
        "burnin_length": 2,        # burn-in期間
        "reward_multisteps": 3,    # multistep reward
        "enable_terminal_zero_reward": True,

        # その他
        "processor": processor,
        "memory": PERRankBaseMemory(
            capacity= 100_000,
            alpha=0.8,           # PERの確率反映率
            beta_initial=0.0,    # IS反映率の初期値
            beta_steps=5_000,  # IS反映率の上昇step数
            enable_is=True,     # ISを有効にするかどうか
        ),
        #"demo_memory": PERProportionalMemory(100_000, alpha=0.8),
        "demo_episode_dir": episode_save_dir,
        "demo_ratio_initial": 1.0,
        "demo_ratio_final": 1/256.0,
        "demo_ratio_steps": 5_000,

        "episode_memory": PERProportionalMemory(600, alpha=0.8),
        "episode_ratio": 1.0/32.0,
    }

    env.close()
    return kwargs



def run_rainbow(enable_train):
    kwargs = create_parameter()
    kwargs["train_interval"] = 1
    kwargs["action_policy"] = AnnealingEpsilonGreedy(
        initial_epsilon=1.0,     # 初期ε
        final_epsilon=0.01,      # 最終状態でのε
        exploration_steps=5_000  # 初期→最終状態になるまでのステップ数
    )

    run_gym_rainbow(enable_train, ENV_NAME, kwargs,
        nb_steps=20_000,
        log_interval1=2000,
        is_load_weights=False,
        skip_movie_save=True,
    )
    

class MyActor(Actor):
    def getPolicy(self, actor_index, actor_num):
        return EpsilonGreedy(0.1)

    def fit(self, index, agent):
        env = gym.make(ENV_NAME)
        agent.fit(env, visualize=False, verbose=0)
        env.close()

class MyActor1(MyActor):
    def getPolicy(self, actor_index, actor_num):
        return EpsilonGreedy(0.01)

class MyActor2(MyActor):
    def getPolicy(self, actor_index, actor_num):
        return EpsilonGreedy(0.1)

def run_r2d3(enable_train):
    kwargs = create_parameter()

    kwargs["actors"] = [MyActor1]
    #kwargs["actors"] = [MyActor1, MyActor2]
    kwargs["gamma"] = 0.997
    kwargs["actor_model_sync_interval"] = 50  # learner から model を同期する間隔

    run_gym_r2d3(enable_train, ENV_NAME, kwargs,
        nb_trains=20_000,
        test_actor=MyActor,
        log_warmup=0,
        log_interval1=10,
        log_interval2=20,
        log_change_count=5,
        log_test_episodes=10,
        is_load_weights=False,
        skip_movie_save=True
    )



if __name__ == '__main__':
    kwargs = create_parameter()
    env = gym.make(ENV_NAME)
    
    #run_play(env, episode_save_dir, kwargs["processor"])
    #run_replay(episode_save_dir)

    run_rainbow(enable_train=True)
    #run_rainbow(enable_train=False)  # test only

    #run_r2d3(enable_train=True)
    #run_r2d3(enable_train=False)  # test only

