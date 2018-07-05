import pickle
import time

import MeCab
from my_tools import sql_posi_nega
from variable_setup import ms, vn

# ========
# 変数の定義
# ========
directory = ''
training_name = 'p01'  # model 00 (~ 99)
train_N, test_N, e, c, H, O, epoch_size = ms[training_name]


checkpoint = time.time()


def check(s=''):
    global checkpoint
    prev_cp = checkpoint
    checkpoint = time.time()
    print('checkpoint:', round(checkpoint - prev_cp, 3), '\n', s)


def prepare(n):  # n must be dividable by 2
    posi, nega = sql_posi_nega(n)
    p_l = list(posi['memo'])
    n_l = list(nega['memo'])

    return p_l + n_l


def mecab_to_vec(string):  # 文を単語に分解して返す。その際、単語を辞書に登録。
    # '表層形\t品詞,品詞細分類1,品詞細分類2,品詞細分類3,活用型,活用形,原形,読み,発音'
    _mecab = MeCab.Tagger("-Ochasen")
    s = string.replace('"', '\'')  # 文字列の区切りがわからなくなるエラー防止
    parsed = _mecab.parse(s)
    # 最後EOSと無駄な空行で2つ1要素の行ができる。それをスライスで取り除く。
    res = {s.split('\t')[2] for s in parsed.split('\n')[:-2]}
    return res


if __name__ == '__main__':
    print(training_name)
    N = train_N + test_N
    all_reviews = prepare(N)
    print('review size:', len(all_reviews))

    check('mecab_to_vec all of the reviews')
    vocab_set = {'<SOS>', '<EOS>'}
    batch = 500
    for i in range(0, len(all_reviews), batch):
        reviews = all_reviews[i:i + batch]
        check('set' + str(i))
        for s in reviews:
            vocab_set |= mecab_to_vec(s)

    vocab = {}
    for i, w in enumerate(vocab_set):
        if i + 1 % 10000 == 0:
            check('dic' + str(i))
        vocab[w] = i

    with open('' + vn[training_name] + '.pickle', 'wb') as f:
        pickle.dump(vocab, f)

    print(len(vocab))
