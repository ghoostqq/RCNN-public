# ========
# import
# ========
# basic libraries
import glob
import pickle
import random
from datetime import datetime as dt

# Chainer
import chainer
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from chainer import serializers

# my module.
from RecurrentCNN import RecurrentCNN
from variable_setup import ms, xy  # module settings

matplotlib.use('Agg')  # ディスプレイがないサーバ上でもplotできるように。

# ========
# 変数の定義
# ========
directory = ''
training_name = 'm11'  # model 00 (~ 99)
train_N, test_N, e, c, H, O, epoch_size = ms[training_name]


def np_float(num_lst):
    return np.array(num_lst).astype(np.float32)


def day_time_detailed():
    return dt.now().strftime('%m/%d_%H:%M:%S')


def prepare():
    with open('' + xy[training_name] + '.pickle', 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    # 1
    # chainerの準備
    # データ準備
    N = train_N + test_N
    all_X, all_Y, V, SOS, EOS = prepare()

    # 訓練データとテストデータを分ける。
    X = all_X[:train_N]
    Y = all_Y[:train_N]
    tX = all_X[train_N:]
    tY = all_Y[train_N:]

    # あるモデルで、あるドキュメント集合の判定精度（％）を算出し、リストに追加
    def accu_list_append(model, accu_list, tX, tY):
        ok = 0
        for xt, yt in zip(tX, tY):
            with chainer.no_backprop_mode():
                res = model.fwd(xt)
            ans = res.data
            if np.argmax(ans) == np.argmax(yt):
                ok += 1

        # print(ok, '/', test_N, ' = ', ok / test_N)
        accu_list.append(ok / test_N)

    # list up learned models
    trained_models = glob.glob(
        directory + training_name + '_[0-9]*_e[0-9]*.model')
    # calc accuracy for each models
    traX, traY = zip(*random.sample(list(zip(X, Y)), 100))
    train_acc = []
    test_acc = []
    for model_file in sorted(trained_models):
        print(model_file[-21:])
        # calc accu.
        # Setup Model
        model = RecurrentCNN(V, e, c, H, O, SOS, EOS)
        # Load the Model.
        serializers.load_npz(model_file, model)
        # Test model accuracy.
        print('train_acc', day_time_detailed())
        accu_list_append(model, train_acc, traX, traY)
        print('test_acc', day_time_detailed())
        accu_list_append(model, test_acc, tX, tY)
    print(test_acc)
    print(mean(test_acc))

    try:
        plt.figure(figsize=(5.5, 4.5), dpi=144)
        # plt.plot(test_acc)
        # plt.savefig(directory+training_name+'_acc_test.png')
        # plt.plot(train_acc)
        # plt.savefig(directory+training_name+'_acc_train.png')
        plt.plot(test_acc)
        plt.plot(train_acc)
        plt.savefig(directory + training_name + '_acc_tes_tra.png')
    except Exception:
        print('plot error')
