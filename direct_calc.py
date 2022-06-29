import json

import numpy as np
from sklearn.metrics.pairwise import paired_distances


# 单纯计算距离小于0.8的为一类的朴素方法
if __name__ == '__main__':
    with open('feature_json/face_features_json.json', 'r', encoding="utf-8") as f:
        face_dict = json.load(f)

    print(len(face_dict))
    entered = set()
    cluster_res = []
    for k, v in face_dict.items():
        if k not in entered:
            entered.add(k)
            res = {k}
            for k1, v1 in face_dict.items():
                if (k1 in res) or (k1 in entered):
                    continue
                else:
                    if paired_distances(np.asarray([v1]), np.asarray([v]), metric='cosine') < 0.6:
                        res.add(k1)
                        entered.add(k1)
            cluster_res.append(res.copy())

    print(list(map(lambda x: len(x), cluster_res)))
    print(len(list(map(lambda x: len(x), cluster_res))))