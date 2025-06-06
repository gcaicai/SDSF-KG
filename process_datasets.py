""" Reconstruct train:valid:test sets of three datasets at the ratio of 8:1:1 for embedding training. """

import os
import numpy as np
from collections import Counter, defaultdict as ddict
import pickle
import random

random.seed(12345)


def load_data(path):
    if 'FB13' in path:
        ent_path = 'entity2id.txt'
        rel_path = 'relation2id_new.txt'
    else:
        ent_path = 'entity2id_new.txt'
        rel_path = 'relation2id.txt'

    # get idx-->entity
    with open(os.path.join(path, ent_path), errors='ignore') as f:
        entity2id = dict()
        lst = f.readlines()  # list[str]
        del lst[0]
        for line in lst:
            ent_id = line.strip('\n').split('\t')
            entity2id[int(ent_id[1])] = ent_id[0]  # key(int)-->value(str)

    # get idx-->relation
    with open(os.path.join(path, rel_path), errors='ignore') as f:
        relation2id = dict()
        lst = f.readlines()  # list[str]
        del lst[0]
        for line in lst:
            rel_id = line.strip('\n').split('\t')
            relation2id[int(rel_id[1])] = rel_id[0]

    train_triples = read_triples(os.path.join(path, 'train2id.txt'))
    valid_triples = read_triples(os.path.join(path, 'valid2id.txt'))
    test_triples = read_triples(os.path.join(path, 'test2id.txt'))

    return entity2id, relation2id, train_triples, valid_triples, test_triples


def read_triples(path):
    triples = []
    if 'FB13' in path:
        with open(path) as f:
            lst = f.readlines()
            del lst[0]
            for line in lst:
                head, tail, relation = line.strip('\n').split('\t')
                triples.append((int(head), int(relation), int(tail)))
    else:
        with open(path) as f:
            lst = f.readlines()
            del lst[0]
            for line in lst:
                head, tail, relation = line.strip('\n').split(' ')
                triples.append((int(head), int(relation), int(tail)))
    return np.array(triples)


if __name__ == '__main__':
    file_read_path = './datasets/FB15K237/'  # file read path, choices = ['FB13', 'FB15K', 'FB15K237']
    file_write_path = './datasets/FB15K237/FB15K237.pkl'  # storage path, choices = ['/FB13/FB13.pkl', '/FB15K/FB15K.pkl', '/FB15K237/FB15K237.pkl']

    entity2id, relation2id, train_triples, valid_triples, test_triples = load_data(file_read_path)

    # integrate train set, valid set, test set
    triples = np.concatenate((train_triples, valid_triples), axis=0)
    triples = np.concatenate((triples, test_triples), axis=0)

    np.random.shuffle(triples)

    # count the frequency of entities and relations
    ent_count = Counter(h for h, _, _ in triples) + Counter(t for _, _, t in triples)
    rel_count = Counter(r for _, r, _ in triples)

    ent_freq = ddict(int, ent_count)
    rel_freq = ddict(int, rel_count)

    # reconstruct train:valid:test sets at the ratio of 8:1:1
    train_triples = []
    valid_triples = []
    test_triples = []

    for idx, tri in enumerate(triples):
        h, r, t = tri
        # if the frequency of h r t >=2 , classify them into valid and test sets, else train set
        if ent_freq[h] > 2 and ent_freq[t] > 2 and rel_freq[r] > 2:
            test_triples.append(tri)
            ent_freq[h] -= 1
            ent_freq[t] -= 1
            rel_freq[r] -= 1
        else:
            train_triples.append(tri)

        if len(test_triples) > int(len(triples) * 0.2):
            train_triples.extend(triples[idx + 1:])
            break

    random.shuffle(test_triples)
    test_len = len(test_triples)

    valid_triples = test_triples[:int(test_len / 2)]  # valid set
    test_triples = test_triples[int(test_len / 2):]  # test set

    train_edge_index = np.array(train_triples)[:, [0, 2]].T
    train_edge_type = np.array(train_triples)[:, 1].T

    valid_edge_index = np.array(valid_triples)[:, [0, 2]].T
    valid_edge_type = np.array(valid_triples)[:, 1].T

    test_edge_index = np.array(test_triples)[:, [0, 2]].T
    test_edge_type = np.array(test_triples)[:, 1].T

    data_dict = {'train': {'edge_index': train_edge_index, 'edge_type': train_edge_type},
                 'test': {'edge_index': test_edge_index, 'edge_type': test_edge_type},
                 'valid': {'edge_index': valid_edge_index, 'edge_type': valid_edge_type}}

    pickle.dump(data_dict, open(file_write_path, 'wb'))
