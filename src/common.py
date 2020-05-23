import tensorflow as tf
from keras import backend as K
import numpy as np

import enum
import math
import os
import random


def clipped_error_loss(y_true, y_pred):
    err = y_true - y_pred  # エラー
    L2 = 0.5 * K.square(err)
    L1 = K.abs(err) - 0.5

    # エラーが[-1,1]区間ならL2、それ以外ならL1を選択する。
    loss = tf.where((K.abs(err) < 1.0), L2, L1)   # Keras does not cover where function in tensorflow :-(
    return K.mean(loss)

def rescaling(x, epsilon=0.001):
    n = math.sqrt(abs(x)+1) - 1
    return np.sign(x)*n + epsilon*x


class InputType(enum.Enum):
    VALUES = 1    # 画像無し
    GRAY_2ch = 3  # (width, height)
    GRAY_3ch = 4  # (width, height, 1)
    COLOR = 5     # (width, height, ch)

class LstmType(enum.Enum):
    NONE = 0
    STATELESS = 1
    STATEFUL = 2

class DuelingNetwork(enum.Enum):
    AVERAGE = 0
    MAX = 1
    NAIVE = 2


class LoggerType(enum.Enum):
    TIME = 1
    STEP = 2

# copy from https://qiita.com/okotaku/items/8d682a11d8f2370684c9
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    #session_conf = tf.compat.v1.ConfigProto(
    #    intra_op_parallelism_threads=1,
    #    inter_op_parallelism_threads=1
    #)
    #sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
    #tf.compat.v1.keras.backend.set_session(sess)

