"""
Experiment for AROC algorithm.
"""
import json
import time

import numpy as np

from aroc import aroc
from metrics import f1_score


def main():
    with open('feature_json/face_features_json.json', 'r', encoding="utf-8") as f:
        face_dict = json.load(f)

    features = np.asarray(list(face_dict.values()))
    label_lookup = dict()
    person_stat = dict()
    for k, i in enumerate(face_dict.keys()):
        label_lookup[k] = int(i.split("_")[1])
        if int(i.split("_")[1]) in person_stat:
            person_stat[int(i.split("_")[1])] += 1
        else:
            person_stat[int(i.split("_")[1])] = 1
    print("人数统计", person_stat)
    start_time = time.time()
    clusters = aroc(features, 40, 0.5, 60)
    print('Time taken for clustering: {:.3f} seconds'.format(
        time.time() - start_time))

    print(clusters)
    _, _, _, precision, recall, score = f1_score(
        clusters, label_lookup)
    print('Clusters: {}  Precision: {:.3f}  Recall: {:.3f}  F1: {:.3f}'.format(
        len(clusters), precision, recall, score))


if __name__ == '__main__':
    main()
