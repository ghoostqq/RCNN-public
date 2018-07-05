# ========
# import
# ========
# basic libraries
import pickle

import chainer
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from chainer import Variable, optimizers, serializers

# my module.
from my_tools import Lap, day_time_detailed, day_time_str
from RecurrentCNN import RecurrentCNN  # import my model.
from variable_setup import ms, xy  # module settings

matplotlib.use('Agg')  # ディスプレイがないサーバ上でもplotできるように。


# ========
# 変数の定義
# ========
directory = ''
training_name = 'm12'  # model 00 (~ 99)
print(training_name)
train_N, test_N, e, c, H, O, epoch_size = ms[training_name]


def prepare():
    with open('' + xy[training_name] + '.pickle', 'rb') as f:
        return pickle.load(f)


# # #######################################
# # Restrict recource
# _rsrc = resource.RLIMIT_AS
# _soft, _hard = resource.getrlimit(_rsrc)
#
# _k = 1024
# _soft = _k**4
# resource.setrlimit(_rsrc, (_soft,_hard) )
#
# # #######################################

# 1
# chainerの準備
# データ準備
print('Prepare Datas.')
N = train_N + test_N
print('N =', N)
all_X, all_Y, V, SOS, EOS = prepare()

# 訓練データとテストデータを分ける。
X = all_X[:train_N]
Y = all_Y[:train_N]
tX = all_X[train_N:]
tY = all_Y[train_N:]


# 2
# ################################
# use imported RecurrentCNN model.
# ################################


# 3
print('Setup Model.')
model = RecurrentCNN(V, e, c, H, O, SOS, EOS)
optimizer = optimizers.SGD()
optimizer.setup(model)
# Weight decay
optimizer.add_hook(chainer.optimizer.WeightDecay(0.0005))


# 4
print('Training Start.', day_time_detailed())
loss_list = []
epoch_lap = Lap()
for epoch in range(epoch_size):
    # 経過epoch数を表示
    print('=== epoch:', epoch, '=== |', day_time_detailed(), end='')
    epoch_lap.time('e', show=True)
    # データの加工
    # 訓練データを一通りfor文で学習にかける
    for i in range(train_N):
        x = X[i]
        y = Variable(Y[i])

        model.zerograds()
        loss = model(x, y)
        # 節目のエポックで訓練前後の損失関数を表示
        if epoch % 5 == 4 and (i == 0 or i == train_N - 1):
            print(loss.data)
        # 一通り学習した段階で損失関数を保存
        if (i == 0 or i == train_N - 1):
            loss_list.append(loss.data)
        loss.backward()
        optimizer.update()

    # 途中保存
    # 一通り学習したところで、エポックがキリのいいところで保存する
    if epoch % 5 == 4:
        out_file = training_name + '_' + day_time_str() + '_e' + str(epoch) + '.model'
        # out_f='{0}_e{1}_{2}.model'.format(training_name,epoch,day_time_str())
        # 名前形式が変わるので簡単に変えられない。
        serializers.save_npz(directory + out_file, model)


# ======================================
# 5
accu_list = []
ok = 0
for i in range(test_N):
    xt = tX[i]
    yt = tY[i]
    with chainer.no_backprop_mode():
        res = model.fwd(xt)
    ans = res.data
    if np.argmax(ans) == np.argmax(yt):
        ok += 1

print(ok, '/', test_N, ' = ', ok / test_N)
# accu_list.append( ok / test_N )
# print(accu_list)
# ======================================

# print(loss_list)
with open('' + training_name + '_loss.pickle', 'wb') as f:
    pickle.dump(loss_list, f)

try:
    plt.figure(figsize=(5.5, 4.5), dpi=144)
    plt.plot(loss_list)
    plt.savefig(directory + training_name + '_loss_' + day_time_str() + '.png')
except Exception:
    print('plot error')
