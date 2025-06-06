""" This file aims to standardize the presentation of entities and relations across the FB13, FB15K, FB15K237 datasets.
    The mapping values are sourced from the following website:
        https://github.com/zhw12/BERTRL/blob/master/data/text/FB237/entity2text.txt
"""
import re


def read_source_file(file_path):
    map_info_lst = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            map_index, info = line.strip().split('\t')
            map_info_lst.append([map_index, info.lower()])
    return map_info_lst


def read_mapping_file(file_path):
    node2id_lst = []
    with open(file_path, 'r') as file:
        file.readline()
        for line in file:
            node, idx = line.strip().split('\t')
            node2id_lst.append(node.lower())
    return node2id_lst


def replace(map_info_lst, ent2id_lst):
    map_lst = list(list(zip(*map_info_lst))[0])
    intersection = list(set(map_lst).intersection(set(ent2id_lst)))

    for index, item in enumerate(ent2id_lst):
        if item in intersection:
            locate = map_lst.index(item)
            map_info = map_info_lst[locate][1]
            special_char = re.findall(r'[^a-zA-Z0-9\s\-_]', map_info)  # uniform special characters
            if ' ' in map_info:
                map_info = map_info.replace(' ', '_')
            if len(special_char) > 0:
                for char in special_char:
                    map_info = map_info.replace(char, '_')
            # replace '__' as '_'
            map_info = re.sub(r'_{2,}', '_', map_info)

            ent2id_lst[index] = map_info

    return ent2id_lst


def reconstruct_rel(source_rel, target_rel):
    source_last_rel = [str.split('/')[-1] for str in source_rel]

    target_rel_lst = [
        next((source_rel[idx] for idx, sour_str in enumerate(source_last_rel) if tar_str in sour_str), tar_str)
        for tar_str in target_rel
    ]

    return target_rel_lst


if __name__ == '__main__':
    """ uniform content in entity2id.txt
        transform entity2id.txt of FB15K, FB15K-237 as the same as FB13.
    """
    source_ent_path = './datasets/maps/FB237_entity2text.txt'
    mapping_ent_path = './datasets/FB15K237/entity2id.txt'  # FB15K237
    # mapping_ent_path = './datasets/FB15K/entity2id.txt'  # FB15K

    map_info_lst = read_source_file(source_ent_path)
    ent2id_lst = read_mapping_file(mapping_ent_path)
    ent2id_lst_new = replace(map_info_lst, ent2id_lst)

    write_path = './datasets/FB15K237/entity2id_new.txt'  # FB15K237
    # write_path = './datasets/FB15K/entity2id_new.txt'  # FB15K

    with open(write_path, "w", encoding='utf-8') as file:
        file.write(str(len(ent2id_lst_new)) + "\n")
        for index, row in enumerate(ent2id_lst_new):
            file.write(row + '\t' + str(index) + "\n")

    """ uniform content in relation2id.txt
        transform relation2id.txt of FB13 as the same as FB15K, FB15K-237.
    """
    source_rel_path = './datasets/FB15K237/relation2id.txt'
    target_rel_path = './datasets/FB13/relation2id.txt'
    write_rel_path = './datasets/FB13/relation2id_new.txt'

    source_rel_lst = read_mapping_file(source_rel_path)
    target_rel_lst = read_mapping_file(target_rel_path)

    target_rel_new = reconstruct_rel(source_rel_lst, target_rel_lst)

    with open(write_rel_path, "w", encoding='utf-8') as file:
        file.write(str(len(target_rel_new)) + "\n")
        for index, row in enumerate(target_rel_new):
            file.write(row + '\t' + str(index) + "\n")
