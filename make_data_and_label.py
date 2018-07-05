import pickle
import random

import numpy as np

import MeCab
from my_tools import sql_posi_nega
from variable_setup import ms, vn, xy

# ========
# 変数の定義
# ========
directory = ''
training_name = 'p01'  # model 00 (~ 99)
train_N, test_N, e, c, H, O, epoch_size = ms[training_name]


def np_int(num_lst):
    return np.array(num_lst).astype(np.int32)


def np_float(num_lst):
    return np.array(num_lst).astype(np.float32)


def make_data_and_label(n):  # n must be dividable by 2
    posi, nega = sql_posi_nega(n)
    p_l = list(posi['memo'])
    n_l = list(nega['memo'])
    random.seed(777)
    random.shuffle(p_l)
    random.shuffle(n_l)
    reviews = [p_l[i // 2] if i % 2 == 0 else n_l[i // 2] for i in range(n)]

    def mecab_to_vec(string):  # 文を単語に分解して返す。その際、単語を辞書に登録。
        # '表層形\t品詞,品詞細分類1,品詞細分類2,品詞細分類3,活用型,活用形,原形,読み,発音'
        p = {'名詞', '形容詞', '副詞'}
        _mecab = MeCab.Tagger("-Ochasen")
        s = string.replace('"', '\'')  # 文字列の区切りがわからなくなるエラー防止
        # parsed = _mecab.parse(s)
        node = _mecab.parseToNode(s)
        res = []
        while node:
            n = node.feature.split(',')
            if n[0] in p:
                # print(n[0], n[6])
                res.append(n[6])
            node = node.next
        # 最後EOSと無駄な空行で2つ1要素の行ができるが、品詞で排除できる。
        # 品詞でフィルタした後文の要素数が0になった場合、<NUL>を追加する。
        if len(res) == 0:
            res.append('<NUL>')
        return res

    reviews_word = [mecab_to_vec(s) for s in reviews]
    vocab_set = {'<SOS>', '<EOS>', '<NUL>'}
    batch = 200
    for i in range(0, len(reviews_word), batch):
        reviews = reviews_word[i:i + batch]
        for s in reviews:
            vocab_set |= set(s)

    vocab = {}
    for i, w in enumerate(vocab_set):
        vocab[w] = i

    with open('' + vn[training_name] + '.pickle', 'wb') as f:
        pickle.dump(vocab, f)

    reviews_vec = [np_int([vocab[w] for w in r]) for r in reviews_word]

    label = [np_float([[1, 0]]), np_float([[0, 1]])] * (n // 2)

    return (reviews_vec, label, len(vocab), vocab['<SOS>'], vocab['<EOS>'])


print(training_name)
N = train_N + test_N
dataset = make_data_and_label(N)
with open('' + xy[training_name] + '.pickle', 'wb') as f:
    pickle.dump(dataset, f)

all_X, all_Y, V, SOS, EOS = dataset
print(len(all_X), len(all_Y), V, SOS, EOS)
