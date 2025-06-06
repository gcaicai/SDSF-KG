""" mutual information (MI) + Adversarial inference success rate (AISR) """
import pickle
import numpy as np
from collections import Counter
import torch
import random
import copy
import math


policy_threshold = 0.01


def with_access(vic_attr, adver_attr, vic_triples, adver_triples):
    # adversary knowledge: triples related to d attributes of the sender (not considering the whole sharing subgraph)
    adver_triple = []
    adver_ent = random.sample(vic_attr, math.ceil(policy_threshold * len(adver_attr)))

    entities = [h for h, _, t in vic_triples] + [t for h, _, t in vic_triples]
    ent_counter = Counter(entities)
    sorted_counter = ent_counter.most_common(int(0.001 * len(ent_counter)))  # filter high frequency entities of sender
    adver_counter = [ent for ent, _ in sorted_counter]
    adver_ent.extend(adver_counter)

    for tri in adver_triples:
        if tri[0] in adver_ent or tri[2] in adver_ent:
            adver_triple.append(tri)

    return adver_triple


def without_access(vic_attr, adver_attr, adver_triples):
    # adversary knowledge: triples related to (d-1) attributes of the sender
    adver_ent = random.sample(vic_attr, math.floor(policy_threshold * len(adver_attr)))
    adver_triple = [tri for tri in adver_triples if tri[0] in adver_ent or tri[2] in adver_ent]

    return adver_triple


def get_leak_triple(num, graph):
    triples = []
    count = 0
    new_graph = copy.deepcopy(graph)
    new_graph = random.sample(new_graph, len(new_graph))
    for subgraph in new_graph:
        if count <= num:
            for item in list(subgraph.edges(data=True)):
                h, t, r = item[0], item[1], item[2]['relation']
                triples.append([h, r, t])
            count += subgraph.number_of_edges()
        else:
            break
    return triples


def get_subgraph_triples(subgraph):
    triples = []
    for item in list(subgraph.edges(data=True)):
        h, t, r = item[0], item[1], item[2]['relation']
        triples.append([h, r, t])
    return triples


def get_triples(path):
    data = pickle.load(open(path, 'rb'))

    train_triples = np.stack((data['train']['edge_index'][0],
                              data['train']['edge_type'],
                              data['train']['edge_index'][1])).T

    valid_triples = np.stack((data['valid']['edge_index'][0],
                              data['valid']['edge_type'],
                              data['valid']['edge_index'][1])).T

    test_triples = np.stack((data['test']['edge_index'][0],
                             data['test']['edge_type'],
                             data['test']['edge_index'][1])).T

    triples = np.concatenate([train_triples, valid_triples, test_triples])

    return triples


def calculate_entropy(triples):
    """ calculate entropy: H(X) """
    entities = [h for h, _, t in triples] + [t for h, _, t in triples]
    triples = [tuple(triple) for triple in triples]

    triple_counter = Counter(triples)
    triple_len = len(triples)
    P_triple = [count / triple_len for count in triple_counter.values()]

    entity_counter = Counter(entities)
    entity_len = len(entities)
    P_entity = [count / entity_len for count in entity_counter.values()]

    H_triple = -sum(p * np.log2(p) for p in P_triple if p > 0)
    H_entity = -sum(p * np.log2(p) for p in P_entity if p > 0)

    return H_triple, H_entity


def calculate_joint_entropy(vic_triples, adver_triples):
    """ calculate joint entropy: H(X,Y) """
    joint_triples = [tuple(tri) for tri in vic_triples + adver_triples]
    triple_counter_joint = Counter(joint_triples)
    triple_len = len(joint_triples)

    triple_joint_probabilities = [count / triple_len for count in triple_counter_joint.values()]
    triple_joint_entropy = -sum(p * np.log2(p) for p in triple_joint_probabilities if p > 0)

    joint_entities = [h for h, _, t in joint_triples] + [t for h, _, t in joint_triples]
    entity_counter_joint = Counter(joint_entities)
    entity_len = len(joint_entities)

    entity_joint_probabilities = [count / entity_len for count in entity_counter_joint.values()]
    entity_joint_entropy = -sum(p * np.log2(p) for p in entity_joint_probabilities if p > 0)

    return triple_joint_entropy, entity_joint_entropy


def calculate_mutual_information(vic_triples, adver_triples):
    """ calculate mutual information: I(X;Y) = H(X) + H(Y) - H(X,Y) """
    # calculate entropy: H(X), H(Y)
    H_triple_vic, H_entity_vic = calculate_entropy(vic_triples)
    H_triple_adver, H_entity_adver = calculate_entropy(adver_triples)

    # calculate joint entropy: H(X,Y)
    joint_triple, joint_entity = calculate_joint_entropy(vic_triples, adver_triples)

    # calculate MI: I(X;Y) = H(X) + H(Y) - H(X,Y)
    I_triple = H_triple_vic + H_triple_adver - joint_triple
    I_entity = H_entity_vic + H_entity_adver - joint_entity

    alpha = I_triple / (I_triple + I_entity)
    beta = I_entity / (I_triple + I_entity)

    mutual_information = alpha * I_triple + beta * I_entity

    return mutual_information


if __name__ == '__main__':
    # whether the receiver is authorized for satisfying access policy, default False
    authorized = True  # Unauth.|datasets or Auth.|datasets

    compare_state = False  # whether the original data is leaked # With|x%
    leak_rate = 0.01  # choices = ['0.01', '0.1', ...]

    # FB15K237 is the receiver
    receiver_subgraph_path = './datasets/FB15K237/subgraph.pkl'
    receiver_attribute_path = './datasets/FB15K237/attribute.pkl'
    receiver_subgraph = pickle.load(open(receiver_subgraph_path, 'rb'))
    receiver_attribute = pickle.load(open(receiver_attribute_path, 'rb'))
    receiver_idx = 4  # the biggest subgraph of FB15K237

    # FB13 is the sender which is attacked
    sender_subgraph_path = './datasets/FB13/subgraph.pkl'  # choices = ['FB13', 'FB15K', ...]
    sender_attribute_path = './datasets/FB13/attribute.pkl'
    sender_triple_path = './datasets/FB13/FB13.pkl'
    sender_subgraph = pickle.load(open(sender_subgraph_path, 'rb'))
    sender_attribute = pickle.load(open(sender_attribute_path, 'rb'))
    sender_idx = 1  # the biggest subgraph of FB13
    sender_triples = get_triples(sender_triple_path)

    receiver_subgraph_triples = get_subgraph_triples(receiver_subgraph[receiver_idx])  # get receiver's biggest subgraph
    sender_subgraph_triples = get_subgraph_triples(sender_subgraph[sender_idx])  # get sender's biggest subgraph

    # get adversary knowledge of triples under different settings
    if authorized:
        adversary_triples = with_access(sender_attribute[sender_idx], receiver_attribute[receiver_idx],
                                        sender_subgraph_triples, receiver_subgraph_triples)
    else:
        adversary_triples = without_access(sender_attribute[sender_idx], receiver_attribute[receiver_idx],
                                           receiver_subgraph_triples)

    # whether With|x%
    if compare_state:
        need_num = leak_rate * len(sender_triples) - len(adversary_triples)
        new_leak_triple = get_leak_triple(need_num, sender_subgraph)
        adversary_triples.extend(new_leak_triple)

    ''' calculate mutual information (MI) '''
    MI = calculate_mutual_information(sender_triples.tolist(), adversary_triples)
    print("MI = ", MI)


    ''' calculate adversarial inference success rate (AISR) '''
    if authorized:
        adver_data = './datasets/FB15K237/FB15K237_FB13.pkl'  # choices = ['FB15K237_FB13.pkl', 'FB15K237_FB15K.pkl', ...]
        adver_rank = './state/FB15K237-FB13-TransE.rank'  # choices = ['FB15K237-FB13-TransE.rank', 'FB15K237-FB15K-TransE.rank', ...]
    else:
        adver_data = './datasets/FB15K237/FB15K237.pkl'  # you can choose the specified files
        adver_rank = './state/FB15K237-TransE.rank'

    adver_data = pickle.load(open(adver_data, 'rb'))
    adver_rank = torch.load(adver_rank).tolist()
    top_1 = [item[1] for item in Counter(adver_rank).most_common(1)][0]  # get the top 1 rank entity of adversary

    # get adversary knowledge triples
    test_triples = np.stack((adver_data['test']['edge_index'][0],
                             adver_data['test']['edge_type'],
                             adver_data['test']['edge_index'][1])).T
    test_triples_f2 = test_triples[:, :2].tolist()

    # if unauthorized, using tail entity in .rank file or top 1 rank entity to infer (h,r,?)
    infer_triples = []
    if not authorized:
        triple_in_test = []
        triple_notin_test = []
        for tri in adversary_triples:
            if tri[:2] in test_triples_f2:
                triple_in_test.append(tri)
            else:
                triple_notin_test.append(tri)
        for tri in triple_in_test:
            idx = test_triples_f2.index(tri[:2])
            infer_triples.append([tri[0], tri[1], adver_rank[idx]])
        for tri in triple_notin_test:
            infer_triples.append([tri[0], tri[1], top_1])
    # if authorized, finding tail entity in sharing subgraph or the same way as unauthorized to infer (h,r,?)
    else:
        triple_in_share = sender_subgraph_triples
        triple_notin_share = adversary_triples
        triple_in_share_f2 = [tri[:2] for tri in triple_in_share]
        triple_in_test = []
        triple_notin_test = []
        for tri in triple_notin_share:
            if tri[:2] in triple_in_share_f2:
                idx = triple_in_share_f2.index(tri[:2])
                infer_triples.append([tri[0], tri[1], triple_in_share[idx][2]])
            elif tri[:2] in test_triples_f2:
                triple_in_test.append(tri)
            else:
                triple_notin_test.append(tri)
        if len(triple_in_test) != 0:
            for tri in triple_in_test:
                idx = test_triples_f2.index(tri[:2])
                infer_triples.append([tri[0], tri[1], adver_rank[idx]])
        if len(triple_notin_test) != 0:
            for tri in triple_notin_test:
                infer_triples.append([tri[0], tri[1], top_1])

    infer_total_num = len(sender_triples)

    infer_arr = np.array(infer_triples)
    correct_infer_count = np.isin(infer_arr.tolist(), sender_triples.tolist()).all(axis=1).sum()

    if authorized:
        print("AISR under Auth.|dataset = ", correct_infer_count/(infer_total_num - len(sender_subgraph_triples)))
    else:
        print("AISR under Unauth.|dataset = ", correct_infer_count/infer_total_num)
