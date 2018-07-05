import time
from datetime import datetime as dt

import numpy as np
import pandas.io.sql as psql

import MySQLdb


# ========
# utility
# ========
def day_time_str():
    return dt.now().strftime('%m%d%H%M')


def day_time_detailed():
    return dt.now().strftime('%H:%M .%S')  # '%m/%d_%H:%M:%S'


class Lap:
    """docstring for Lap ."""

    def __init__(self):
        self.data = {}
        # self.data = defaultdict(lambda: 0)
        self.cp = time.time()

    def prog_bar(self, n):
        sec = int(n)
        hur = sec / (60 * 60)
        fmn = sec / (60 * 5)
        mnt = sec / (60)
        return 'H' * int(hur) + '*' * int(fmn) + '.' * int(mnt)

    def time(self, lap_name='unclassified', show=False):
        prev_cp = self.cp
        self.cp = time.time()
        this_lap = self.cp - prev_cp
        self.data[lap_name] = self.data.get(lap_name, 0) + this_lap
        # self.data[lap_name] += this_lap
        if show:
            print(self.prog_bar(this_lap))


def np_int(num_lst):
    return np.array(num_lst).astype(np.int32)


def np_float(num_lst):
    return np.array(num_lst).astype(np.float32)


def sql_posi_nega(n):
    _conn = MySQLdb.connect(user='',
                            passwd='',
                            host='',
                            db='',
                            charset='utf8')

    def mysql(query):
        df = psql.read_sql(query, _conn)
        return df

    # ランダムのシード値を設定してある。
    # _query = '''SELECT memo
    #             FROM review JOIN evaluation ON review.post_id = evaluation.post_id
    #             WHERE yesuseful - nouseful {condi}
    #             ORDER BY RAND(777)
    #             LIMIT {lmt};'''
    _query = '''SELECT memo
                FROM review JOIN evaluation ON review.post_id = evaluation.post_id
                WHERE {condi}
                LIMIT {lmt};'''
    # _query = '''SELECT memo
    #             FROM review JOIN evaluation ON review.post_id = evaluation.post_id
    #             WHERE evaluate7 {cal}
    #             ORDER BY RAND(777)
    #             LIMIT {lmt};'''
    posi = mysql(_query.format(
        condi='nouseful = 0 ORDER BY yesuseful DESC, RAND(777)', lmt=n // 2))
    nega = mysql(_query.format(
        condi='yesuseful = 0 ORDER BY nouseful DESC, RAND(777)', lmt=n // 2))
    return posi, nega
