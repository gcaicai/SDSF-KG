""" experiments with SDSF-KG (w/ SDSF-KG)
    merge sharing sub-KGs for KGE training on link prediction and triple classification tasks
"""

import pickle
import numpy as np
from collections import Counter, defaultdict as ddict
import random
import itertools


def get_node2id(file_path):
    node2id_lst = []
    with open(file_path, 'r') as file:
        file.readline()
        for line in file:
            node, idx = line.strip().split('\t')
            node2id_lst.append(node.lower())
    return node2id_lst


def get_subgraph_triple(subgraph):
    subgraph_triple = []  # get subgraph triples
    for item in list(subgraph.edges(data=True)):
        h, t, r = item[0], item[1], item[2]['relation']
        subgraph_triple.append([h, r, t])
    return subgraph_triple


def reconstruct_triple(ent_sender_path, rel_sender_path, ent_receiver_path, rel_receiver_path, subgraph_triple, triple_old):
    """ reconstruct triples index after sharing for further KGE training """
    ent_sender = get_node2id(ent_sender_path)  # entity plaintext
    rel_sender = get_node2id(rel_sender_path)  # relation plaintext

    ent_receiver = get_node2id(ent_receiver_path)
    rel_receiver = get_node2id(rel_receiver_path)

    ent_subgraph_idx = list(set(tri[i] for tri in subgraph_triple for i in (0, 2)))  # entity index
    rel_subgraph_idx = list(set(tri[1] for tri in subgraph_triple))  # relation index

    ent_subgraph = [ent_sender[idx] for idx in ent_subgraph_idx]  # entity plaintext
    rel_subgraph = [rel_sender[idx] for idx in rel_subgraph_idx]  # relation plaintext

    ent_intersection = set(ent_subgraph).intersection(set(ent_receiver))  # entity plaintext
    rel_intersection = set(rel_subgraph).intersection(set(rel_receiver))  # relation plaintext

    # filter and eliminate parts of unrelated triples which does not promote KGE training
    filtered_triple = []
    for idx, tri in enumerate(subgraph_triple):
        # filter (h,r,?) or (?,r,t) in (h,r,t)
        if ((ent_sender[tri[0]] in ent_intersection and rel_sender[tri[1]] in rel_intersection) or
                (ent_sender[tri[2]] in ent_intersection and rel_sender[tri[1]] in rel_intersection)):
            filtered_triple.append(tri)

    # reconstruct triple index, already existing triple index not change, new added triple index should reconstruct
    ent_subgraph_idx = list(set(tri[i] for tri in filtered_triple for i in (0, 2)))
    rel_subgraph_idx = list(set(tri[1] for tri in filtered_triple))

    ent_subgraph = [ent_sender[idx] for idx in ent_subgraph_idx]
    rel_subgraph = [rel_sender[idx] for idx in rel_subgraph_idx]

    ent_subgraph_recon = []
    rel_subgraph_recon = []

    recon_idx = len(ent_receiver)  # reconstruct entity index
    for ent in ent_subgraph:
        if ent in ent_intersection:
            ent_subgraph_recon.append(ent_receiver.index(ent))
        else:
            ent_subgraph_recon.append(recon_idx)
            recon_idx += 1

    recon_idx = len(rel_receiver)  # reconstruct relation index
    for rel in rel_subgraph:
        if rel in rel_intersection:
            rel_subgraph_recon.append(rel_receiver.index(rel))
        else:
            rel_subgraph_recon.append(recon_idx)
            recon_idx += 1

    # update triple index
    for idx, tri in enumerate(filtered_triple):
        filtered_triple[idx][0] = ent_subgraph_recon[ent_subgraph_idx.index(tri[0])]
        filtered_triple[idx][1] = rel_subgraph_recon[rel_subgraph_idx.index(tri[1])]
        filtered_triple[idx][2] = ent_subgraph_recon[ent_subgraph_idx.index(tri[2])]

    ''' re-process receiver's training data as 8/1/1 '''
    train_head = triple_old['train']['edge_index'][0]
    train_tail = triple_old['train']['edge_index'][1]
    train_relation = triple_old['train']['edge_type']
    train_triple = [[train_head[i], train_relation[i], train_tail[i]] for i in range(len(train_relation))]

    valid_head = triple_old['valid']['edge_index'][0]
    valid_tail = triple_old['valid']['edge_index'][1]
    valid_relation = triple_old['valid']['edge_type']
    valid_triple = [[valid_head[i], valid_relation[i], valid_tail[i]] for i in range(len(valid_relation))]

    test_head = triple_old['test']['edge_index'][0]
    test_tail = triple_old['test']['edge_index'][1]
    test_relation = triple_old['test']['edge_type']
    test_triple = [[test_head[i], test_relation[i], test_tail[i]] for i in range(len(test_relation))]

    # merge old triple + filtered triple
    all_new_triple = list(itertools.chain(train_triple, valid_triple, test_triple, filtered_triple))
    random.shuffle(all_new_triple)

    # count frequency of entity and relation
    ent_count = Counter(h for h, _, _ in all_new_triple) + Counter(t for _, _, t in all_new_triple)
    rel_count = Counter(r for _, r, _ in all_new_triple)

    ent_freq = ddict(int, ent_count)
    rel_freq = ddict(int, rel_count)

    # reconstruct train:valid:test sets at the ratio of 8:1:1
    train_triples = []
    valid_triples = []
    test_triples = []

    for idx, tri in enumerate(all_new_triple):
        h, r, t = tri
        # if the frequency of h r t >=2 , classify them into valid and test sets, else train set
        if ent_freq[h] > 2 and ent_freq[t] > 2 and rel_freq[r] > 2:
            test_triples.append(tri)
            ent_freq[h] -= 1
            ent_freq[t] -= 1
            rel_freq[r] -= 1
        else:
            train_triples.append(tri)

        if len(test_triples) > int(len(all_new_triple) * 0.2):
            train_triples.extend(all_new_triple[idx + 1:])
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

    triple_new = {'train': {'edge_index': train_edge_index, 'edge_type': train_edge_type},
                  'test': {'edge_index': test_edge_index, 'edge_type': test_edge_type},
                  'valid': {'edge_index': valid_edge_index, 'edge_type': valid_edge_type}}
    return triple_new


if __name__ == '__main__':
    sender_path = './datasets/FB13/'  # choices = ['FB13', 'FB15K', ...]

    # FB13 is the sender
    if 'FB13' in sender_path:
        ent2id_sender_path = sender_path + 'entity2id.txt'
        rel2id_sender_path = sender_path + 'relation2id_new.txt'
    else:
        ent2id_sender_path = sender_path + 'entity2id_new.txt'
        rel2id_sender_path = sender_path + 'relation2id.txt'
    subgraph_sender_path = sender_path + 'subgraph.pkl'

    target_id = 1  # index of the biggest subgraph
    data = pickle.load(open(subgraph_sender_path, 'rb'))
    target_triple = get_subgraph_triple(data[target_id])

    # FB15K237 is the receiver, default value
    ent2id_receiver_path = './datasets/FB15K237/entity2id_new.txt'
    rel2id_receiver_path = './datasets/FB15K237/relation2id.txt'

    data_path_old = './datasets/FB15K237/FB15K237.pkl'  # original triples of the receiver
    data_old = pickle.load(open(data_path_old, 'rb'))

    data_new = reconstruct_triple(ent2id_sender_path, rel2id_sender_path, ent2id_receiver_path, rel2id_receiver_path,
                                  target_triple, data_old)  # get new triples after reconstructing and splitting

    if 'FB13' in sender_path:
        data_path_new = './datasets/FB15K237/FB15K237_FB13.pkl'
    else:
        data_path_new = './datasets/FB15K237/FB15K237_FB15K.pkl'

    pickle.dump(data_new, open(data_path_new, 'wb'))

