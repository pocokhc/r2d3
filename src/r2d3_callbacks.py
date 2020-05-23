import rl.callbacks
import matplotlib.pyplot as plt
import numpy as np

import os
import time
import glob
import json
import tempfile

from .r2d3 import R2D3Callback, R2D3
from .common import LoggerType


class SaveManager(R2D3Callback):
    def __init__(self,
            save_dirpath,
            is_load=False,
            save_overwrite=True,
            save_memory=False,
            checkpoint=False,
            checkpoint_interval=10000,
            verbose=1,
        ):
        self.save_dirpath = save_dirpath
        self.is_load = is_load
        self.save_overwrite = save_overwrite
        self.save_memory = save_memory
        self.checkpoint = checkpoint
        self.checkpoint_interval = checkpoint_interval
        self.verbose = verbose

    def on_r2d3_learner_begin(self, learner):
        if not self.is_load:
            return
        path = os.path.join(self.save_dirpath, "last", "learner.dat")
        if self.verbose > 0:
            print("load: {}".format(path))
        learner.load_weights(path, load_memory=self.save_memory)

    def on_r2d3_learner_train_end(self, learner):
        if not self.checkpoint:
            return
        n = learner.train_count.value
        if (n+1) % self.checkpoint_interval == 0:
            dirname = self._get_checkpoint_dir(n+1)
            path = os.path.join(dirname, "learner.dat")
            if self.verbose > 0:
                print("save: {}".format(path))
            learner.save_weights(path, overwrite=True, save_memory=self.save_memory)

    def on_r2d3_learner_end(self, learner):
        dirname = os.path.join(self.save_dirpath, "last")
        os.makedirs(dirname, exist_ok=True)
        path = os.path.join(dirname, "learner.dat")
        if self.verbose > 0:
            print("save: {}".format(path))
        learner.save_weights(path, self.save_overwrite, save_memory=self.save_memory)

    def on_r2d3_actor_begin(self, index, actor):
        self.actor = actor
        if not self.is_load:
            return
        path = os.path.join(self.save_dirpath, "last", "actor{}.dat".format(index))
        if self.verbose > 0:
            print("load: {}".format(path))
        actor.load_weights(path)

    def on_r2d3_actor_end(self, index, actor):
        dirname = os.path.join(self.save_dirpath, "last")
        os.makedirs(dirname, exist_ok=True)
        path = os.path.join(dirname, "actor{}.dat".format(index))
        if self.verbose > 0:
            print("save: {}".format(path))
        actor.save_weights(path, self.save_overwrite)

    def on_step_end(self, episode, logs={}):
        if not self.checkpoint:
            return
        n = self.actor.train_count.value
        if (n+1) % self.checkpoint_interval == 0:
            dirname = self._get_checkpoint_dir(n+1)
            path = os.path.join(dirname, "actor{}.dat".format(self.actor.actor_index))
            if self.verbose > 0:
                print("save: {}".format(path))
            self.actor.save_weights(path, overwrite=True)

    def _get_learner_path(self):
        return os.path.join(self.save_dirpath, "last", "learner.dat")

    def _get_checkpoint_dir(self, count):
        dirname = os.path.join(self.save_dirpath, "checkpoint_{}".format(count))
        os.makedirs(dirname, exist_ok=True)
        path = os.path.join(dirname)
        return path



class Logger2StageR2D3(R2D3Callback):
    def __init__(self,
            interval1,  # stage1 の interval
            interval2,  # stage2 の interval
            change_count,  # stage1->2 へ移行する回数
            warmup=0,      # warmupの時間(この時間まで計測しない)
            savedir="",    # 保存出力先
            test_actor=None,  # testで使うActorクラス
            test_env=None,    # testで使うenv
            test_episodes=10,  # testで実行するエピソード回数
            verbose=1
        ):
        self.savedir = savedir
        self.warmup = warmup
        self.interval1 = interval1
        self.interval2 = interval2
        self.change_count = change_count
        self.test_actor = test_actor
        self.test_env = test_env
        self.test_episodes = test_episodes
        self.verbose = verbose
        
    def _add_logfile(self, filename, data):
        path = os.path.join(self.savedir, filename)
        with open(path, "a") as f:
            f.write("{}\n".format(json.dumps(data)))

    def _is_record(self):
        if self.stage == 1:
            if time.time() - self.t1 < self.warmup:
                return False
            self.stage = 2
        elif self.stage == 2:
            if time.time() - self.t1 < self.interval1:
                return False
            self.stage_count += 1
            if self.stage_count > self.change_count:
                self.stage = 3
        elif self.stage == 3:
            if time.time() - self.t1 < self.interval2:
                return False
        
        self.t1 = time.time()
        return True

    def on_r2d3_train_begin(self):
        os.makedirs(self.savedir, exist_ok=True)
        for fn in glob.glob(os.path.join(self.savedir, "*.json")):
            os.remove(fn)
        
        self.t0 = self.t1 = time.time()
        self.stage_count = 0
        if self.warmup == 0:
            self.stage = 2
        else:
            self.stage = 1

    def on_r2d3_train_end(self):
        if self.verbose > 0:
            print("done, took {:.3f} minutes".format((time.time() - self.t0)/60.0))

    #--- learner ---

    def _record_learner(self, learner):
        d = {
            "name": "learner",
            "time": time.time() - self.t0,
            "train_count": learner.train_count.value,
        }
        if self.test_actor is not None:
            with tempfile.TemporaryDirectory() as tmpdir:
                name = os.path.join(tmpdir, "tmp")
                learner.save_weights(name, overwrite=True)
                test_agent = R2D3.createTestAgentStatic(learner.kwargs, self.test_actor, name)
            history = test_agent.test(self.test_env, nb_episodes=self.test_episodes, visualize=False, verbose=False)
            rewards = np.asarray(history.history["episode_reward"])

            d["test_reward_min"] = float(rewards.min())
            d["test_reward_ave"] = float(rewards.mean())
            d["test_reward_med"] = float(np.median(rewards))
            d["test_reward_max"] = float(rewards.max())
        else:
            d["test_reward_min"] = 0
            d["test_reward_ave"] = 0
            d["test_reward_med"] = 0
            d["test_reward_max"] = 0
        
        self._add_logfile("learner.json", d)
        if self.verbose > 0:
            m = d["time"] / 60.0
            print("{:8} Train {}, Time: {:.2f}m, TestReward: {:7.2f} - {:7.2f} (ave: {:7.2f}, med: {:7.2f})".format(
                d["name"],
                d["train_count"],
                m,
                d["test_reward_min"],
                d["test_reward_max"],
                d["test_reward_ave"],
                d["test_reward_med"]))
    
    def on_r2d3_learner_train_end(self, learner):
        if not self._is_record():
            return
        self._record_learner(learner)

    def on_r2d3_learner_end(self, learner):
        self._record_learner(learner)

    #--- actor ---

    def _actor_init(self):
        self.rewards = []
        self.actor_count = 0

    def _record_actor(self, index, logs={}):
        if len(self.rewards) == 0:
            self.rewards = [0]
        rewards = np.asarray(self.rewards)
        d = {
            "name": "actor{}".format(index),
            "time": time.time() - self.t0,
            "reward_min": float(rewards.min()),
            "reward_ave": float(rewards.mean()),
            "reward_med": float(np.median(rewards)),
            "reward_max": float(rewards.max()),
            "count": self.actor_count,
            "train_count": int(self.actor.train_count.value),
            "nb_steps": int(logs.get("nb_steps", 0)),
        }
        self._actor_init()

        if self.verbose > 0:
            m = d["time"] / 60.0
            print("{:8} Train {}, Time: {:.2f}m, Reward    : {:7.2f} - {:7.2f} (ave: {:7.2f}, med: {:7.2f}), nb_steps: {}".format(
                d["name"],
                d["train_count"],
                m,
                d["reward_min"],
                d["reward_max"],
                d["reward_ave"],
                d["reward_med"],
                d["nb_steps"]))
        self._add_logfile("actor{}.json".format(self.actor_index), d)

    def on_r2d3_actor_begin(self, index, actor):
        self.actor_index = index
        self.actor = actor
    
    def on_r2d3_actor_end(self, index, actor):
        self._record_actor(index)

    def on_train_begin(self, logs={}):
        self._actor_init()

    def on_episode_end(self, episode, logs={}):
        self.rewards.append(logs["episode_reward"])
        self.actor_count += 1
        
        if not self._is_record():
            return
        self._record_actor(self.actor_index, logs)

    #--- other ---

    def getLogs(self):
        logs = []

        for fn in glob.glob(os.path.join(self.savedir, "*.json")):
            with open(fn, "r") as f:
                for line in f:
                    d = json.loads(line)
                    logs.append(d)

        return logs

    def drawGraph(self, base="time", actors=-1):
        
        learner_logs = {
            "x": [], 
            "ax2_y": [], 
            "y1": [], 
            "y2": [], 
            "y3": [], 
            "y4": []
        }
        actors_logs = {}
        x_max = 0
        for log in self.getLogs():
            name = log["name"]

            t = log["time"]/60.0
            if base == "time":
                if x_max < t:
                    x_max = t
            else:
                if x_max < log["train_count"]:
                    x_max = log["train_count"]
            
            if name == "learner":
                if base == "time":
                    learner_logs["x"].append(t)
                    learner_logs["ax2_y"].append(log["train_count"])
                else:
                    learner_logs["x"].append(log["train_count"])
                    learner_logs["ax2_y"].append(t)
                learner_logs["y1"].append(log["test_reward_min"])
                learner_logs["y2"].append(log["test_reward_ave"])
                learner_logs["y3"].append(log["test_reward_med"])
                learner_logs["y4"].append(log["test_reward_max"])

            else:
                if name not in actors_logs:
                    actors_logs[name] = {
                        "x": [], 
                        "ax2_y": [], 
                        "y1": [], 
                        "y2": [], 
                        "y3": [], 
                        "y4": []
                    }
                if base == "time":
                    actors_logs[name]["x"].append(t)
                    actors_logs[name]["ax2_y"].append(log["train_count"])
                else:
                    actors_logs[name]["x"].append(log["train_count"])
                    actors_logs[name]["ax2_y"].append(t)

                actors_logs[name]["y1"].append(log["reward_min"])
                actors_logs[name]["y2"].append(log["reward_ave"])
                actors_logs[name]["y3"].append(log["reward_med"])
                actors_logs[name]["y4"].append(log["reward_max"])
        
        if actors == -1:
            n = len(actors_logs)+1
        else:
            n = actors + 1

        # learner
        fig = plt.figure()
        ax1 = fig.add_subplot(n, 1, 1)
        ax2 = ax1.twinx()
        ax2.plot(learner_logs["x"], learner_logs["ax2_y"], color="black", linestyle="dashed")
        ax1.plot(learner_logs["x"], learner_logs["y1"], marker="o", label="min")
        ax1.plot(learner_logs["x"], learner_logs["y2"], marker="o", label="ave")
        ax1.plot(learner_logs["x"], learner_logs["y3"], marker="o", label="med")
        ax1.plot(learner_logs["x"], learner_logs["y4"], marker="o", label="max")
        ax1.set_xlim([0, x_max])
        ax1.grid(True)
        ax1.legend()
        if base == "time":
            ax1.set_title("Time(m)")
            ax2.set_ylabel("TrainCount")
        else:
            ax1.set_title("TrainCount")
            ax2.set_ylabel("Time(m)")
        ax1.set_ylabel("Learner")

        # actors
        for i in range(n-1):
            name = "actor{}".format(i)
            v = actors_logs[name]

            ax1 = fig.add_subplot(n, 1, 2+i)
            ax2 = ax1.twinx()
            ax2.plot(v["x"], v["ax2_y"], color="black", linestyle="dashed")
            ax1.plot(v["x"], v["y1"], marker="o", label="min")
            ax1.plot(v["x"], v["y2"], marker="o", label="ave")
            ax1.plot(v["x"], v["y3"], marker="o", label="med")
            ax1.plot(v["x"], v["y4"], marker="o", label="max")
            ax1.set_xlim([0, x_max])
            ax1.grid(True)
            ax1.set_ylabel(name)
        
        plt.show()

