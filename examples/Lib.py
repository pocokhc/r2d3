import gym
from keras.optimizers import Adam

import traceback

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from src.r2d3 import R2D3
from src.rainbow import Rainbow
from src.callbacks import Logger2Stage, ModelIntervalCheckpoint, MovieLogger, ConvLayerView
from src.r2d3_callbacks import Logger2StageR2D3, SaveManager
from src.env_play import EpisodeSave, EpisodeReplay
from src.common import InputType, LoggerType


def run_gym_rainbow(
        enable_train,
        env_name,
        kwargs, 
        nb_steps,
        log_interval1=2000,
        log_interval2=0,
        log_change_count=10,
        enable_test=True,
        log_test_episodes=10,
        is_load_weights=False,
        checkpoint_interval=0,
        skip_movie_save=False,
    ):
    if log_interval2 == 0:
        log_interval2 = log_interval1

    env = gym.make(env_name)

    weight_file = "tmp/{}_rainbow_weight.h5".format(env_name)
    os.makedirs(os.path.dirname(weight_file), exist_ok=True)

    agent = Rainbow(**kwargs)
    print(agent.model.summary())

    if enable_test:
        test_agent = Rainbow(**kwargs)
        test_env = gym.make(env_name)
    else:
        test_agent = None
        test_env = None
    log = Logger2Stage(LoggerType.STEP,
        warmup=kwargs["memory_warmup_size"],
        interval1=log_interval1,
        interval2=log_interval2,
        change_count=log_change_count,
        savefile="tmp/{}_log.json".format(env_name),
        test_agent=test_agent,
        test_env=test_env,
        test_episodes=log_test_episodes,
    )

    if enable_train:
        print("--- start ---")
        print("'Ctrl + C' is stop.")
        try:
            callbacks = [log]
            if is_load_weights:
                agent.load_weights(weight_file, load_memory=True)

            if checkpoint_interval > 0:
                callbacks.append(
                    ModelIntervalCheckpoint(
                        filepath = weight_file + '_{step:02d}.h5',
                        interval=checkpoint_interval,
                        save_memory=False,
                    )
                )

            agent.fit(env, nb_steps=nb_steps, visualize=False, verbose=0, callbacks=callbacks)
            test_env.close()

        except Exception:
            print(traceback.print_exc())

        # save
        print("weight save: " + weight_file)
        agent.save_weights(weight_file, overwrite=True, save_memory=False)
        
    # plt
    log.drawGraph("step")

    # 訓練結果を見る
    print("weight load: " + weight_file)
    agent.load_weights(weight_file)
    agent.test(env, nb_episodes=5, visualize=True)

    # 動画保存用
    if not skip_movie_save:
        movie = MovieLogger()
        callbacks = [movie]
        if agent.input_type != InputType.VALUES:
            conv = ConvLayerView(agent)
            callbacks.append(conv)
        agent.test(env, nb_episodes=1, visualize=False, callbacks=callbacks)
        movie.save(gifname="tmp/{}_1.gif".format(env_name), fps=30)
        if agent.input_type != InputType.VALUES:
            conv.save(grad_cam_layers=["conv_1", "conv_2", "conv_3"], add_adv_layer=True, add_val_layer=True, 
                end_frame=200, gifname="tmp/{}_2.gif".format(env_name), fps=10)

    env.close()



def run_gym_r2d3(
        enable_train,
        env_name,
        kwargs,
        nb_trains,
        test_actor=None,
        log_warmup=0,
        log_interval1=10,
        log_interval2=0,
        log_change_count=5,
        log_test_episodes=10,
        is_load_weights=False,
        checkpoint_interval=0,
        skip_movie_save=False,
    ):
    if log_interval2 == 2:
        log_interval2 = log_interval1
    env = gym.make(env_name)

    # R2D3
    manager = R2D3(**kwargs)

    if test_actor is None:
        test_env = None
    else:
        test_env = gym.make(env_name)
    log = Logger2StageR2D3(
        warmup=log_warmup,
        interval1=log_interval1,
        interval2=log_interval2,
        change_count=log_change_count,
        savedir="tmp_{}".format(env_name),
        test_actor=test_actor,
        test_env=test_env,
        test_episodes=log_test_episodes,
    )

    if enable_train:
        print("--- start ---")
        print("'Ctrl + C' is stop.")
        save_manager = SaveManager(
            save_dirpath="tmp_{}".format(env_name),
            is_load=is_load_weights,
            save_memory=True,
            checkpoint=(checkpoint_interval>0),
            checkpoint_interval=checkpoint_interval,
            verbose=0
        )

        manager.train(nb_trains=nb_trains, callbacks=[save_manager, log])

    # plt
    log.drawGraph("train")

    # 訓練結果を見る
    agent = manager.createTestAgent(kwargs["actors"][0], "tmp_{}/last/learner.dat".format(env_name))
    agent.test(env, nb_episodes=5, visualize=True)

    # 動画保存用
    if not skip_movie_save:
        movie = MovieLogger()
        callbacks = [movie]
        if kwargs["input_type"] != InputType.VALUES:
            conv = ConvLayerView(agent)
            callbacks.append(conv)
        agent.test(env, nb_episodes=1, visualize=False, callbacks=callbacks)
        movie.save(gifname="tmp/{}_1.gif".format(env_name), fps=30)
        if kwargs["input_type"] != InputType.VALUES:
            conv.save(grad_cam_layers=["conv_1", "conv_2", "conv_3"], add_adv_layer=True, add_val_layer=True, 
                end_frame=200, gifname="tmp/{}_2.gif".format(env_name), fps=10)

    env.close()



def run_play(env, episode_save_dir, processor, **kwargs):
    es = EpisodeSave(env, 
        episode_save_dir=episode_save_dir,
        processor=processor,
        font="arial")
    es.play(**kwargs)
    env.close()


def run_replay(episode_save_dir, **kwargs):
    r = EpisodeReplay(episode_save_dir, font="arial")
    r.play(**kwargs)




