from keras.models import model_from_json
from keras.optimizers import Adam
import numpy as np

import random
import math

from .common import LstmType, clipped_error_loss


class Policy():
    """ Abstract base class for all implemented Policy. """

    def compile(self, model_json):
        pass

    def get_weights(self):
        pass
    
    def set_weights(self, params):
        pass

    def select_action(self, agent):
        raise NotImplementedError()
    

class EpsilonGreedy(Policy):
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def select_action(self, agent):
        if self.epsilon > random.random():
            # アクションをランダムに選択
            action = random.randint(0, agent.nb_actions-1)
        else:
            # 評価が最大のアクションを選択
            action = np.argmax(agent.get_qvals())
        return action


class EpsilonGreedyActor(EpsilonGreedy):
    def __init__(self, actor_index, actors_length, epsilon=0.4, alpha=7):
        if actors_length <= 1:
            tmp = epsilon ** (1 + alpha)
        else:
            tmp = epsilon ** (1 + actor_index/(actors_length-1)*alpha)
        super().__init__(epsilon=tmp)


class AnnealingEpsilonGreedy(Policy):
    """ native dqn pilocy
    https://arxiv.org/abs/1312.5602
    """

    def __init__(self,  
            initial_epsilon=1,  # 初期ε
            final_epsilon=0.1,  # 最終状態でのε
            exploration_steps=1_000_000  # 初期→最終状態になるまでのステップ数
        ):
        self.epsilon_step = (initial_epsilon - final_epsilon) / exploration_steps
        self.initial_epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.step = 0

    def get_weights(self):
        return {
            "step": self.step
        }
    
    def set_weights(self, params):
        self.step = params["step"]

    def select_action(self, agent):

        # epsilon の計算
        epsilon = self.initial_epsilon - self.step * self.epsilon_step
        if epsilon < self.final_epsilon:
            epsilon = self.final_epsilon
        self.step += 1

        if epsilon > random.random():
            # アクションをランダムに選択
            action = random.randint(0, agent.nb_actions-1)
        else:
            # 評価が最大のアクションを選択
            action = np.argmax(agent.get_qvals())
        return action


class SoftmaxPolicy(Policy):

    def select_action(self, agent):
        qvals = agent.get_qvals()
        exp_x = np.exp(qvals)

        vals = []
        for i in range(agent.nb_actions):
            # softmax 値以下の乱数を生成
            vals.append( random.uniform(0, exp_x[i]) )

        # 乱数の結果一番大きいアクションを選択
        action = np.argmax(vals)
        return action

class UCB1(Policy):
    def compile(self, model_json):
        self.count_model = model_from_json(model_json)
        self.count_model.compile(loss=clipped_error_loss, optimizer=Adam())
        
    def get_weights(self):
        return {
            "count": self.count_model.get_weights()
        }
    
    def set_weights(self, params):
        self.count_model.set_weights(params["count"])

    def select_action(self, agent):
        assert agent.lstm_type != LstmType.STATEFUL, "no support LSTMFUL"

        observation = agent.get_state()
        state = observation[np.newaxis,:]

        counts = self.count_model.predict(state, 1)[0]
        qvals = agent.get_qvals()

        # counts は1以上にする
        for i in range(len(counts)):
            if counts[i] < 1:
                counts[i] = 1

        total = sum(counts)
        total = math.log(total)  # 重そうな計算なので for の外で

        ucbs = []
        for i in range(agent.nb_actions):
            count = counts[i]
            ave = qvals[i] / count

            tmp = ave + math.sqrt(2 * total / count)
            ucbs.append(tmp)
        
        # ucbが最大値となるアクションを選択
        action = np.argmax(ucbs)

        # 選択したアクションの選択回数を増やす
        counts[action] += 1
        self.count_model.train_on_batch(state, counts[np.newaxis,:])
        
        return action


class UCB1_Tuned(Policy):
    def compile(self, model_json):
        self.count_model = model_from_json(model_json)
        self.count_model.compile(loss=clipped_error_loss, optimizer=Adam())
        self.var_model = model_from_json(model_json)
        self.var_model.compile(loss=clipped_error_loss, optimizer=Adam())
        
    def get_weights(self):
        return {
            "count": self.count_model.get_weights(),
            "var": self.var_model.get_weights(),
        }
    
    def set_weights(self, params):
        self.count_model.set_weights(params["count"])
        self.var_model.set_weights(params["var"])

    def select_action(self, agent):
        assert agent.lstm_type != LstmType.STATEFUL, "no support LSTMFUL"

        observation, action, reward = agent.get_prev_state()
        state0 = observation[np.newaxis,:]

        counts = self.count_model.predict(state0, 1)[0]
        ucb_vars = self.var_model.predict(state0, 1)[0]
        qvals = agent.get_qvals()

        # counts は1以上にする
        for i in range(len(counts)):
            if counts[i] < 1:
                counts[i] = 1

        # 分散を更新
        prev_count = counts[action]
        prev_ave = qvals[action] / prev_count
        var = ucb_vars[action]
        var += ((reward - prev_ave) ** 2) / prev_count
        ucb_vars[action] = var
        # 更新
        self.var_model.train_on_batch(state0, np.asarray([ucb_vars]))

        observation = agent.get_state()
        state = observation[np.newaxis,:]

        counts = self.count_model.predict(state, 1)[0]
        qvals = agent.model.predict(state, 1)[0]
        ucb_vars = self.var_model.predict(state, 1)[0]

        # counts は1以上にする
        for i in range(len(counts)):
            if counts[i] < 1:
                counts[i] = 1

        # 分散がマイナスは0以上にする
        for i in range(len(ucb_vars)):
            if ucb_vars[i] < 0:
                ucb_vars[i] = 0

        # 合計を出す(数式ではN)
        total = sum(counts)
        total = math.log(total)  # 重そうな計算なので for の外で

        # 各アクションのUCB値を計算
        ucbs = []
        for i in range(agent.nb_actions):
            count = counts[i]

            # 平均
            ave = qvals[i] / count
            # 分散
            var = ucb_vars[i]

            # 数式を計算
            v = var + math.sqrt(2 * total / count)
            if 1/4 < v:
                v = 1/4
            tmp = ave + math.sqrt( (total / count) * v )
            ucbs.append(tmp)

        # ucbが最大値となるアクションを選択
        action = np.argmax(ucbs)

        # 選択したアクションの選択回数を増やす
        counts[action] += 1
        self.count_model.train_on_batch(state, np.asarray([counts]))

        return action


class UCBv(Policy):
    def compile(self, model_json):
        self.count_model = model_from_json(model_json)
        self.count_model.compile(loss=clipped_error_loss, optimizer=Adam())
        self.var_model = model_from_json(model_json)
        self.var_model.compile(loss=clipped_error_loss, optimizer=Adam())
        
    def get_weights(self):
        return {
            "count": self.count_model.get_weights(),
            "var": self.var_model.get_weights(),
        }
    
    def set_weights(self, params):
        self.count_model.set_weights(params["count"])
        self.var_model.set_weights(params["var"])

    def select_action(self, agent):
        assert agent.lstm_type != LstmType.STATEFUL, "no support LSTMFUL"

        observation, action, reward = agent.get_prev_state()
        state0 = observation[np.newaxis,:]

        counts = self.count_model.predict(state0, 1)[0]
        ucb_vars = self.var_model.predict(state0, 1)[0]
        qvals = agent.get_qvals()

        # counts は1以上にする
        for i in range(len(counts)):
            if counts[i] < 1:
                counts[i] = 1

        # 分散を更新
        prev_count = counts[action]
        prev_ave = qvals[action] / prev_count
        var = ucb_vars[action]
        var += ((reward - prev_ave) ** 2) / prev_count
        ucb_vars[action] = var
        # 更新
        self.var_model.train_on_batch(state0, np.asarray([ucb_vars]))

        observation = agent.get_state()
        state = observation[np.newaxis,:]

        counts = self.count_model.predict(state, 1)[0]
        qvals = agent.model.predict(state, 1)[0]
        ucb_vars = self.var_model.predict(state, 1)[0]

        # counts は1以上にする
        for i in range(len(counts)):
            if counts[i] < 1:
                counts[i] = 1

        # 分散がマイナスは0以上にする
        for i in range(len(ucb_vars)):
            if ucb_vars[i] < 0:
                ucb_vars[i] = 0

        # 合計を出す(数式ではN)
        total = sum(counts)

        # 各アクションのUCB値を計算
        zeta = 1.2
        c = 1
        b = 1
        e = zeta*math.log(total)
        ucbs = []
        for i in range(agent.nb_actions):
            count = counts[i]

            # 平均
            ave = qvals[i] / count
            # 分散
            var = ucb_vars[i]

            tmp = ave + math.sqrt( (2*var*e)/count ) + c* (3*b*e)/count
            ucbs.append(tmp)

        # ucbが最大値となるアクションを選択
        action = np.argmax(ucbs)

        # 選択したアクションの選択回数を増やす
        counts[action] += 1
        self.count_model.train_on_batch(state, np.asarray([counts]))

        return action



class KL_UCB(Policy):
    def __init__(self, C=0, delta=1e-8, eps=1e-12):
        self.C = C
        self.delta = delta  # 探索幅
        self.eps = eps      # 探索の許容誤差

    def compile(self, model_json):
        self.count_model = model_from_json(model_json)
        self.count_model.compile(loss=clipped_error_loss, optimizer=Adam())
        
    def get_weights(self):
        return {
            "count": self.count_model.get_weights(),
        }
    
    def set_weights(self, params):
        self.count_model.set_weights(params["count"])

    def select_action(self, agent):
        assert agent.lstm_type != LstmType.STATEFUL, "no support LSTMFUL"

        observation = agent.get_state()
        state = observation[np.newaxis,:]

        counts = self.count_model.predict(state, 1)[0]
        qvals = agent.get_qvals()

        # counts は1以上にする
        for i in range(len(counts)):
            if counts[i] < 1:
                counts[i] = 1

        # 合計を出す(数式ではN)
        total = sum(counts)

        # 右辺をだしておく
        logndn = math.log(total) + self.C * math.log(math.log(total))
        
        # 各アクションのUCB値を計算
        ucbs = []
        for i in range(agent.nb_actions):
            count = counts[i]
            p = qvals[i] / count

            # 例外処理：p は 0～1
            if p >= 1:
                ucbs.append(1)
                continue
            if p <= 0:
                p = self.delta
            
            # 最大値を探索する
            q = p + self.delta
            converged = False  # debug
            for _ in range(10):
                # kl-divergence
                try:
                    kl = p * math.log(p/q) + (1-p) * math.log((1-p)/(1-q))
                except ValueError:
                    break
                f = logndn - kl
                df = -(q-p)/(q*(1.0-q))

                if f*f < self.eps:
                    converged = True
                    break
                
                q = min([1-self.delta, max([q-f/df, p+self.delta])])

            # debug
            #assert converged, "WARNING:KL-UCB did not converge!! p={} logndn={} q={}".format(p, logndn, q)
            ucbs.append(q)
        
        # ucbが最大値となるアクションを選択
        action = np.argmax(ucbs)

        # 選択したアクションの選択回数を増やす
        counts[action] += 1
        self.count_model.train_on_batch(state, np.asarray([counts]))

        return action

class ThompsonSamplingBeta(Policy):

    def compile(self, model_json):
        self.reward_alpha = model_from_json(model_json)
        self.reward_alpha.compile(loss=clipped_error_loss, optimizer=Adam())
        self.reward_beta = model_from_json(model_json)
        self.reward_beta.compile(loss=clipped_error_loss, optimizer=Adam())
        
    def get_weights(self):
        return {
            "reward_alpha": self.reward_alpha.get_weights(),
            "reward_beta": self.reward_beta.get_weights(),
        }
    
    def set_weights(self, params):
        self.reward_alpha.set_weights(params["reward_alpha"])
        self.reward_beta.set_weights(params["reward_beta"])

    def select_action(self, agent):
        assert agent.lstm_type != LstmType.STATEFUL, "no support LSTMFUL"

        observation, action, reward = agent.get_prev_state()
        state0 = observation[np.newaxis,:]

        # 更新
        if reward > 0:
            v = self.reward_alpha.predict(state0, 1)[0]
            for i in range(len(v)):
                if v[i] < 1:
                    v[i] = 1
            v[action] += 1
            self.reward_alpha.train_on_batch(state0, np.asarray([v]))
        else:
            v = self.reward_beta.predict(state0, 1)[0]
            for i in range(len(v)):
                if v[i] < 1:
                    v[i] = 1
            v[action] += 1
            self.reward_beta.train_on_batch(state0, np.asarray([v]))

        state = agent.get_state()[np.newaxis,:]
        alphas = self.reward_alpha.predict(state, 1)[0]
        betas = self.reward_alpha.predict(state, 1)[0]

        # alpha,beta は1以上にする
        for i in range(len(alphas)):
            if alphas[i] < 1:
                alphas[i] = 1
            if betas[i] < 1:
                betas[i] = 1

        # アクションを計算
        vals = []
        for i in range(agent.nb_actions):

            # ベータ分布に従って乱数を生成
            v = np.random.beta(alphas[i], betas[i])
            vals.append(v)

        # 乱数が最大値となるアクションを選択
        action = np.argmax(vals)
        return action


class ThompsonSamplingGaussian(Policy):
    def __init__(self, dispersion=1):
        # 既知の分散
        self.dispersion = dispersion
        assert self.dispersion != 0

    def compile(self, model_json):
        self.recent_sigma = model_from_json(model_json)
        self.recent_sigma.compile(loss=clipped_error_loss, optimizer=Adam())
        self.recent_mu = model_from_json(model_json)
        self.recent_mu.compile(loss=clipped_error_loss, optimizer=Adam())
        
    def get_weights(self):
        return {
            "recent_sigma": self.recent_sigma.get_weights(),
            "recent_mu": self.recent_mu.get_weights(),
        }
    
    def set_weights(self, params):
        self.recent_sigma.set_weights(params["recent_sigma"])
        self.recent_mu.set_weights(params["recent_mu"])

    def select_action(self, agent):
        assert agent.lstm_type != LstmType.STATEFUL, "no support LSTMFUL"

        observation, action, reward = agent.get_prev_state()
        state0 = observation[np.newaxis,:]

        mu = self.recent_mu.predict(state0, 1)[0]
        sigma = self.recent_sigma.predict(state0, 1)[0]

        # 分散は1以上
        for i in range(len(sigma)):
            if sigma[i] < 1:
                sigma[i] = 1

        # 更新(平均)
        tmp1 = reward/self.dispersion + mu[action]/sigma[action]
        tmp2 = 1/self.dispersion + 1/sigma[action]
        mu[action] = tmp1/tmp2
        self.recent_mu.train_on_batch(state0, np.asarray([mu]))

        # 更新(分散)
        sigma[action] = 1/( (1/self.dispersion) + (1/sigma[action]) )
        self.recent_sigma.train_on_batch(state0, np.asarray([sigma]))

        state = agent.get_state()[np.newaxis,:]
        mu = self.recent_mu.predict(state, 1)[0]
        sigma = self.recent_sigma.predict(state, 1)[0]

        # 分散は1以上
        for i in range(len(sigma)):
            if sigma[i] < 1:
                sigma[i] = 1

        # アクションを計算
        vals = []
        for i in range(agent.nb_actions):
            # 正規分布に従い乱数を生成
            v = np.random.normal(mu[i], sigma[i])
            vals.append(v)

        # 乱数が最大値となるアクションを選択
        action = np.argmax(vals)
        return action


