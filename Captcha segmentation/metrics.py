
'''
定义模型评估指标函数
'''

import numpy as np
from functools import partial, update_wrapper
import keras.backend as K
import pandas as pd

def metric(f):

    def wrapper(y_true, y_pred, *args, **kwargs):
        y_true, y_pred = K.cast(y_true, np.int64), K.cast(y_pred, np.int64)

        if len(y_true.get_shape().as_list()) == 3:
            y_true = K.argmax(y_true, axis=2)

        if len(y_pred.get_shape().as_list()) == 3:
            y_pred = K.argmax(y_pred, axis=2)

        return f(y_true, y_pred, *args, **kwargs)

    update_wrapper(wrapper, f)
    return wrapper


@metric
def char_accuracy(y_true, y_pred):
    '''
    单个字符识别的平均正确率
    '''
    return K.mean(K.cast(K.flatten(K.equal(y_true, y_pred)), np.float32))

@metric
def matchk_accuracy(y_true, y_pred, k=2):
    '''
    至少有k个字符识别正确的平均正确率，k为[1,4]之间的整数，4为图片中所包含的字符数量
    '''
    return K.mean(K.cast(K.greater_equal(K.sum(K.cast(K.equal(y_true, y_pred), np.int64), axis=1), k), np.float32))

@metric
def fullmatch_accuracy(y_true, y_pred):
    '''
    一张图片中四个字符全部识别正确的图片所占的平均比率
    '''
    return K.mean(K.prod(K.cast(K.equal(y_true, y_pred), np.float32), axis=1))


def match1_accuracy(y_true, y_pred):
    return matchk_accuracy(y_true, y_pred, k=1)

def match2_accuracy(y_true, y_pred):
    return matchk_accuracy(y_true, y_pred, k=2)

def match3_accuracy(y_true, y_pred):
    return matchk_accuracy(y_true, y_pred, k=3)

def match4_accuracy(y_true, y_pred):
    return matchk_accuracy(y_true, y_pred, k=4)


def summary(y_true, y_pred):

    metrics = {
        'char_acc': char_accuracy(y_true, y_pred),
        'fullmatch_acc': fullmatch_accuracy(y_true, y_pred)
    }
    for k in range(1, y_true.shape[1]):
        metrics['match{}_acc'.format(k)] = matchk_accuracy(y_true, y_pred, k=k)

    df = pd.DataFrame.from_dict(
        dict([(metric, [round(K.get_value(value), 5)]) for metric, value in metrics.items()] + [('-', 'values')])
    )
    df.set_index(['-'], inplace=True)

    print('Number of samples: {}, Number of characters per sample: {}'.format(*y_true.shape))
    print(df)



