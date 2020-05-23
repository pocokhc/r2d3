# R2D3(Deep Reinforcement Learning) for Keras-RL
以下Qiita記事の実装コードとなります。

+ [【強化学習】R2D3を実装/解説してみた（Keras-RL）](https://qiita.com/pocokhc/items/2e000df3fddddd3d0854)

また、下記記事も参考にしてください。

+ [【強化学習】R2D2を実装/解説してみたリベンジ 解説編（Keras-RL）](https://qiita.com/pocokhc/items/408f0f818140924ad4c4)
+ [【強化学習】R2D2を実装/解説してみたリベンジ ハイパーパラメータ解説編（Keras-RL）](https://qiita.com/pocokhc/items/bc498a1dc720dcf075d6)


# 概要
Keras 向けの強化学習ライブラリである [Keras-rl](https://github.com/keras-rl/keras-rl) の Agent を拡張したものとなります。  
以下のアルゴリズムを実装しています。(非公式です)  

- Rainbow
  - Deep Q Learning (DQN)
  - Double DQN
  - Priority Experience Reply
  - Dueling Network
  - Multi-Step learning
  - (not implemented Noisy Network)
  - (not implemented Categorical DQN)
- Deep Recurrent Q-Learning(DRQN)
- Ape-X
- Recurrent Replay Distributed DQN(R2D2)
- Recurrent Replay Distributed DQN from Demonstrations(R2D3)


# Getting started
## 1. pip install
使っているパッケージは以下です。

+ pip install tensorflow (or tensorflow-cpu or tensorflow-gpu)
+ pip install keras
+ pip install keras-rl
+ pip install gym
+ pip install numpy
+ pip install matplotlib
+ pip install opencv-python
+ pip install pillow
+ pip install pygame

必要に応じて以下のレポジトリも参照してください。

- [OpenAI Gym](https://github.com/openai/gym)
- [Keras-rl](https://github.com/keras-rl/keras-rl)

### 作成時のバージョン

+ windows 10
+ python 3.7.5
+ tensorflow 2.1.0
+ tensorflow-gpu 2.1.0
  + cuda_10.1.243
  + cudnn v7.6.5.32
+ Keras 2.3.1
+ keras-rl 0.4.2
+ gym 0.17.1
+ numpy 1.18.2
+ matplotlib 3.2.1
+ opencv-python 4.1.2.30
+ pillow 6.2.1
+ pygame 1.9.6


## 2. ダウンロード
このレポジトリをダウンロードします。

``` bash
> git clone https://github.com/pocokhc/r2d3.git
> cd r2d3
```

## 3. 実行
examples にいくつか実行例が入っています。

``` bash
> cd r2d3/examples
> python mountaincar.py
```

また、examples のコードはコメントアウトを書き換えることで動作を変更できます。

+ デモ環境プレイ

``` python
if __name__ == '__main__':
    kwargs = create_parameter()
    env = gym.make(ENV_NAME)
    
    run_play(env, episode_save_dir, kwargs["processor"])
    run_replay(episode_save_dir)
``` 

+ rainbow による学習

``` python
if __name__ == '__main__':
    run_rainbow(enable_train=True)
    #run_rainbow(enable_train=False)  # test only
``` 

+ R2D3 による学習

``` python
if __name__ == '__main__':
    run_r2d3(enable_train=True)
    #run_r2d3(enable_train=False)  # test only
```
