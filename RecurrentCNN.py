import chainer.functions as F
import chainer.links as L
import numpy as np
from chainer import Chain


class RecurrentCNN (Chain):
    def __init__(self, V, e, c, H, O, SOS, EOS):
        self.SOS = SOS
        self.EOS = EOS
        super(RecurrentCNN, self).__init__(
            # パラメータを含む関数の宣言（？）
            embed=L.EmbedID(V, e),
            Wl=L.Linear(c, c, nobias=True),
            Wr=L.Linear(c, c, nobias=True),
            Wsl=L.Linear(e, c, nobias=True),
            Wsr=L.Linear(e, c, nobias=True),
            l2=L.Linear(e + 2 * c, H),
            l4=L.Linear(H, O),
        )

    def __call__(self, sentence, label):
        # 損失関数
        y = self.fwd(sentence)
        z = label
        # print(z.data, y.data)
        return F.mean_squared_error(y, z)

    def fwd(self, sentence):
        e_w = [self.embed(np.array([word_id]).astype(np.int32))
               for word_id in sentence]
        # |c| vector Variable.
        cl_w1 = self.embed(np.array([self.SOS]).astype(np.int32))
        cr_wn = self.embed(np.array([self.EOS]).astype(np.int32))
        cl_w = [cl_w1]
        cr_w = [cr_wn]
        n = len(sentence)
        for i in range(0, n - 1):
            # F.n_step_rnn can be used??
            cl_w.append(F.tanh(self.Wl(cl_w[i]) + self.Wsl(e_w[i])))
        _tmp = list(reversed(e_w))
        for i in range(0, n - 1):
            cr_w.append(F.tanh(self.Wr(cr_w[i]) + self.Wsr(_tmp[i])))
        cr_w.reverse()

        xi = [F.concat((cl, e, cr), axis=1)
              for cl, e, cr in zip(cl_w, e_w, cr_w)]
        xi = F.concat(xi, axis=0)
        h2 = F.tanh(self.l2(xi))  # 本当はtanhを入れたいが、なぜか1か-1になってしまう。=> solved!<3

        h3 = F.max(h2, axis=0, keepdims=True)
        h4 = F.softmax(self.l4(h3), axis=1)

        return h4
