""" This file aims to split sub-KGs and extract corresponding attributes (entities) """
import torch
import numpy as np
import hdbscan
from sklearn.preprocessing import StandardScaler
import pickle
import networkx as nx
import community as community_louvain
import random


fine_grained_up = 0.1  # granularity size upper limit
fine_grained_down = 0.01  # granularity size lower limit

top_ratio = 0.2  # select top x% attributes


def hdbscan_cluster(path):
    """ entity clustering (the number of entity is far more than relation) """
    checkpoint = torch.load(path)

    entity_embeddings = checkpoint['ent_emb'].detach().cpu().numpy()  # load entities' embeddings

    scaler = StandardScaler()
    scaled_entity_embeddings = scaler.fit_transform(entity_embeddings)

    # using HDBSCAN to cluster entity embeddings
    hdb = hdbscan.HDBSCAN(min_cluster_size=5, gen_min_span_tree=True)
    hdb_labels = hdb.fit_predict(scaled_entity_embeddings)

    return hdb_labels


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


def split_community_subgraph_second(component):
    """ further split graph by using community_louvain for the second time """
    partition = community_louvain.best_partition(component, resolution=1.2)
    community_num = len(set(partition.values()))

    component_subgraph_list = [nx.MultiGraph() for _ in range(community_num)]

    for item in list(component.edges(data=True)):
        g_h, g_t, g_r = item[0], item[1], item[2]['relation']
        if partition[g_h] == partition[g_t]:
            component_subgraph_list[partition[g_h]].add_edge(g_h, g_t, relation=g_r)
        else:
            if random.randint(0, 1) == 0:
                component_subgraph_list[partition[g_h]].add_edge(g_h, g_t, relation=g_r)
            else:
                component_subgraph_list[partition[g_t]].add_edge(g_h, g_t, relation=g_r)

    return component_subgraph_list


def split_community_subgraph(component, threshold_up):
    """ further split graph by using community_louvain """
    partition = community_louvain.best_partition(component, resolution=1.2)
    community_num = len(set(partition.values()))
    component_subgraph_list = [nx.MultiGraph() for _ in range(community_num)]

    for item in list(component.edges(data=True)):
        g_h, g_t, g_r = item[0], item[1], item[2]['relation']
        if partition[g_h] == partition[g_t]:
            component_subgraph_list[partition[g_h]].add_edge(g_h, g_t, relation=g_r)
        else:
            if random.randint(0, 1) == 0:
                component_subgraph_list[partition[g_h]].add_edge(g_h, g_t, relation=g_r)
            else:
                component_subgraph_list[partition[g_t]].add_edge(g_h, g_t, relation=g_r)

    for item, subgraph in enumerate(component_subgraph_list):
        if subgraph.number_of_edges() > threshold_up:
            sec_subgraph_list = split_community_subgraph_second(subgraph)
            for sec_item, sec_subgraph in enumerate(sec_subgraph_list):
                if sec_item == 0:
                    component_subgraph_list[item] = sec_subgraph_list[sec_item]
                else:
                    component_subgraph_list.append(sec_subgraph)

    return component_subgraph_list


def find_similar_subgraph(small_graph, component):
    """ merge the samll subgraph into the similar one """
    small_graph_nodes = list(small_graph.nodes())
    intersection = 0
    position = -1
    for item, subgraph in enumerate(component):
        subgraph_nodes = list(subgraph.nodes())
        if intersection < len(list(set(small_graph_nodes) & set(subgraph_nodes))):
            intersection = len(list(set(small_graph_nodes) & set(subgraph_nodes)))
            position = item

    if position == -1:
        position = max(range(len(component)), key=lambda i: component[i].number_of_edges())

    return position


def split_subgraph(triples, clusters_ent, threshold_up, threshold_down):
    """ classify triples according to entity clustering results """
    clusters_triples = []
    for i in range(len(np.unique(clusters_ent))):
        clusters_triples.append([])

    for i, triple in enumerate(triples):
        h, r, t = triple
        if clusters_ent[h] == clusters_ent[t]:  # if head and tail in the same cluster
            cluster_idx = clusters_ent[h]
            clusters_triples[cluster_idx].append([h, r, t])
        else:
            if random.randint(0, 1) == 0:
                cluster_idx = clusters_ent[h]
                clusters_triples[cluster_idx].append([h, r, t])
            else:
                cluster_idx = clusters_ent[t]
                clusters_triples[cluster_idx].append([h, r, t])

    num_graphs = len(clusters_triples)

    graph_list = []  # store subgraph

    # turn clusters_triples into nx.multiGraph() in an appropriate size
    for i in range(num_graphs):
        graph = nx.MultiGraph()
        if len(clusters_triples[i]) != 0:
            for j, triple in enumerate(clusters_triples[i]):
                h, r, t = triple
                graph.add_edge(h, t, relation=r)

            # further split graph according to graph connectivity
            connected_components = list(nx.connected_components(graph))
            subgraphs = [graph.subgraph(component).copy() for component in connected_components]

            subgraphs_communities = []

            if len(subgraphs) == 1 and subgraphs[0].number_of_edges() < threshold_up:
                graph_list.append(subgraphs[0])
            else:
                for component in subgraphs:
                    if component.number_of_edges() > threshold_up:
                        # the size of subgraph is too large
                        subgraphs_communities.extend(split_community_subgraph(component, threshold_up))
                    else:
                        subgraphs_communities.append(component)

            graph_list.extend(subgraphs_communities)

    # manage the splitting subgraph
    graph_list_small = []
    graph_list_normal = []
    for idx, component in enumerate(graph_list):
        if component.number_of_edges() != 0:
            if component.number_of_edges() < threshold_down:
                graph_list_small.append(component)
            else:
                graph_list_normal.append(component)

    # merge too small subgraph into the similar one
    for idx, component in enumerate(graph_list_small):
        position = find_similar_subgraph(component, graph_list_normal)
        graph_list_normal[position].update(component)

    return graph_list_normal


def extract_attribute(subgraph_list):
    """ extract top x% intersection of degree centrality and betweenness centrality as attributes """
    subgraph_attribute_list = []

    for i, component in enumerate(subgraph_list):
        degree_centrality = nx.degree_centrality(component)
        betweenness_centrality = nx.betweenness_centrality(component)

        sorted_degree_centrality = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)
        top_x_degree_count = int(len(sorted_degree_centrality) * top_ratio)
        top_x_degree_candidates = {node for node, centrality in sorted_degree_centrality[:top_x_degree_count]}

        sorted_betweenness_centrality = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)
        top_x_betweenness_count = int(len(sorted_betweenness_centrality) * top_ratio)
        top_x_betweenness_candidates = {node for node, centrality in
                                        sorted_betweenness_centrality[:top_x_betweenness_count]}

        # intersection
        representative_nodes = sorted(list(top_x_degree_candidates.intersection(top_x_betweenness_candidates)))
        subgraph_attribute_list.append(representative_nodes)

    return subgraph_attribute_list


if __name__ == '__main__':
    """ split sub-KGs according to hdbscan clustering results of entities,
        then extract vital entities as attributes for each sub-KG
    """
    embedding_path = './state/FB13-TransE.best'  # specify the file path for the target embedding results
    triple_path = './datasets/FB13/FB13.pkl'  # choices = ['FB13', 'FB15K', 'FB15K237']

    attribute_path = './datasets/FB13/attribute.pkl'  # alternative
    subgraph_path = './datasets/FB13/subgraph.pkl'  # alternative

    # using HDBSCAN to cluster entity embeddings
    hdb_labels_entity = hdbscan_cluster(embedding_path)

    # load all triples
    all_triples = get_triples(triple_path)

    threshold_up = int(fine_grained_up * len(all_triples))
    threshold_down = int(fine_grained_down * len(all_triples))

    # split sub-KGs according to the hdbscan results
    subgraph_list = split_subgraph(all_triples, hdb_labels_entity, threshold_up, threshold_down)

    # you can write subgraph information for further processing
    with open(subgraph_path, 'wb') as f:
        pickle.dump(subgraph_list, f)

    # extract attributes from the splitting results
    subgraph_attribute_list = extract_attribute(subgraph_list)

    # you can write attributes for further processing
    with open(attribute_path, 'wb') as f:
        pickle.dump(subgraph_attribute_list, f)

