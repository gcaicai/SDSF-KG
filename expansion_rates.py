""" calculate EER, RER, TER """
import pickle


def get_node2id(path):
    node2id_lst = []
    with open(path, 'r') as file:
        file.readline()
        for line in file:
            node, idx = line.strip().split('\t')
            node2id_lst.append(node.lower())
    return node2id_lst


if __name__ == '__main__':
    # FB15K237 is the receiver
    ent2id_receiver_path = './datasets/FB15K237/entity2id_new.txt'
    rel2id_receiver_path = './datasets/FB15K237/relation2id.txt'
    receiver_path = './datasets/FB15K237/FB15K237.pkl'

    # FB13 is the sender
    sender_path = './datasets/FB13/'  # choices = ['FB13', 'FB15K237', ...]
    if 'FB13' in sender_path:
        ent2id_sender_path = sender_path + 'entity2id.txt'
        rel2id_sender_path = sender_path + 'relation2id_new.txt'
    else:
        ent2id_sender_path = sender_path + 'entity2id_new.txt'
        rel2id_sender_path = sender_path + 'relation2id.txt'
    sender_subgraph_path = sender_path + 'subgraph.pkl'
    sender_idx = 1  # index of the biggest subgraph of FB13
    sender_subgraph = pickle.load(open(sender_subgraph_path, 'rb'))

    sender_triples = []  # get triples of subgraph
    for item in list(sender_subgraph[sender_idx].edges(data=True)):
        g_h, g_t, g_r = item[0], item[1], item[2]['relation']
        sender_triples.append([g_h, g_r, g_t])

    ent_receiver = get_node2id(ent2id_receiver_path)  # plaintext
    rel_receiver = get_node2id(rel2id_receiver_path)

    ent_sender = get_node2id(ent2id_sender_path)  # plaintext
    rel_sender = get_node2id(rel2id_sender_path)

    ent_subgraph_idx = list(set(tri[i] for tri in sender_triples for i in (0, 2)))  # subgraph unique index
    rel_subgraph_idx = list(set(tri[1] for tri in sender_triples))

    ent_subgraph = [ent_sender[idx] for idx in ent_subgraph_idx]  # unique plaintext
    rel_subgraph = [rel_sender[idx] for idx in rel_subgraph_idx]

    ent_intersection = set(ent_subgraph).intersection(set(ent_receiver))  # plaintext
    rel_intersection = set(rel_subgraph).intersection(set(rel_receiver))

    ''' Entity expansion rate (ERR) + Relation expansion rate (RER) '''
    print('ERR = ', round((len(ent_subgraph_idx) - len(ent_intersection)) / len(ent_subgraph_idx) * 100, 4))
    print('RER = ', round((len(rel_subgraph_idx) - len(rel_intersection)) / len(rel_subgraph_idx) * 100, 4))

    ''' Triple enrichment rate (TER) '''
    count = 0
    for idx, tri in enumerate(sender_triples):
        if (ent_sender[tri[0]] in ent_intersection and ent_sender[tri[2]] in ent_intersection and
                rel_sender[tri[1]] in rel_intersection):
            count += 1
    print('TER = ', round((len(sender_triples) - count) / len(sender_triples) * 100, 4))
