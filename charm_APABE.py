""" Implementation of AP-ABE by using Charm-Crypto library """
""" Charm-Crypto
Reference website: https://github.com/JHUISI/charm/blob/dev/charm/schemes/abenc/abenc_yct14.py
Documentation: https://jhuisi.github.io/charm
"""

from charm.toolbox.pairinggroup import PairingGroup, ZR, G1, G2, GT, pair
from charm.toolbox.secretutil import SecretUtil
from charm.toolbox.symcrypto import SymmetricCryptoAbstraction
from charm.toolbox.ABEnc import ABEnc
from charm.core.math.pairing import hashPair as extractor
import pickle
import hashlib


debug = False
policy_threshold = 0.01


def get_ent2id(path):
    ent2id_lst = []
    with open(path, 'r') as file:
        file.readline()
        for line in file:
            ent, idx = line.strip().split('\t')
            ent2id_lst.append(ent.lower())
    return ent2id_lst


def get_subgraph_list(path):
    # load multiGraph
    with open(path, "rb") as file:
        subgraph_list = pickle.load(file)
    return subgraph_list


def get_subgraph_attribute_list(path):
    # load attributes of multiGraph
    with open(path, "rb") as file:
        subgraph_attribute_list = pickle.load(file)
    return subgraph_attribute_list


def get_subgraph_attribute(subgraph_attribute_list, subgraph_idx, ent2id_lst):
    """ calculate hash values for each attribute """
    subgraph_attribute = subgraph_attribute_list[subgraph_idx]

    hash_attribute = []
    for i, item in enumerate(subgraph_attribute):
        plaintext_attribute = ent2id_lst[item]
        hash_attribute.append(hashlib.sha256(plaintext_attribute.encode()).hexdigest())

    return hash_attribute


class APABE(ABEnc):

    def __init__(self, groupObj, verbose=False):
        ABEnc.__init__(self)
        global group, util
        group = groupObj
        util = SecretUtil(group, verbose)

    def setup(self, attribute_sender, attribute_receiver):

        y = group.random(ZR)  # random number
        g = group.random(G1)  # generator
        s = group.random(ZR)  # random number

        enc_attributes_sender = self.__encrypt_attribute(g, s, y, attribute_sender)
        enc_attributes_receiver = self.__encrypt_attribute(g, s, y, attribute_receiver)

        self.attributeSecrets = {}
        self.attribute = {}
        for attri in enc_attributes_sender:
            ti = group.random(ZR)
            self.attributeSecrets[attri] = ti
            self.attribute[attri] = g ** ti
        return g ** y, y, s, enc_attributes_sender, enc_attributes_receiver

    def __encrypt_attribute(self, g, s, y, attributes):
        enc_attributes = []
        for attri in attributes:
            attri_zr = group.hash(attri, ZR)
            result = group.pair(g ** (attri_zr * s + y), g)
            enc_attributes.append(result)
        return enc_attributes

    def keygen(self, pk, mk, policy_str):
        policy = util.createPolicy(policy_str)
        attr_list = util.getAttributeList(policy)

        p = mk
        shares = util.calculateSharesDict(p, policy)

        d = {}
        D = {'policy': policy_str, 'Du': d}
        for x in attr_list:
            y = util.strip_index(x)
            d[y] = shares[x] / self.attributeSecrets[y]
            if debug:
                print(str(y) + " d[y] " + str(d[y]))
        if debug:
            print("Access Policy for key: %s" % policy)
        if debug:
            print("Attribute list: %s" % attr_list)
        return D

    def encrypt(self, pk, s, M, attr_list):
        if debug:
            print('Encryption Algorithm...')
        Es = pk ** s

        Ei = {}
        for attr in attr_list:
            Ei[attr] = self.attribute[attr] ** s

        symcrypt = SymmetricCryptoAbstraction(extractor(Es))
        E = symcrypt.encrypt(M)

        return {'E': E, 'Ei': Ei, 'attributes': attr_list}

    def decrypt(self, E, D):
        policy = util.createPolicy(D['policy'])
        attrs = util.prune(policy, E['attributes'])
        if attrs == False:
            return False
        coeff = util.getCoefficients(policy)

        Z = {}
        prodT = 1
        for i in range(len(attrs)):
            x = attrs[i].getAttribute()
            y = attrs[i].getAttributeAndIndex()
            Z[y] = E['Ei'][x] ** D['Du'][x]
            prodT *= Z[y] ** coeff[y]

        symcrypt = SymmetricCryptoAbstraction(extractor(prodT))

        return symcrypt.decrypt(E['E'])


def kp_abe(sender_path, receiver_path, subgraph_id_sender, subgraph_id_receiver):
    groupObj = PairingGroup('MNT224')
    apabe = APABE(groupObj)

    if 'FB13' in sender_path:
        file_ent2id_sender = sender_path + 'entity2id.txt'
    else:
        file_ent2id_sender = sender_path + 'entity2id_new.txt'
    file_subgraph_attribute_sender = sender_path + 'attribute.pkl'
    file_subgraph_sender = sender_path + 'subgraph.pkl'

    file_ent2id_receiver = './datasets/FB15K237/entity2id_new.txt'
    file_subgraph_attribute_receiver = receiver_path + 'attribute.pkl'

    # obtain attributes of the sender's target subgraph
    ent2id_sender = get_ent2id(file_ent2id_sender)
    subgraph_attribute_list_sender = get_subgraph_attribute_list(file_subgraph_attribute_sender)
    attribute_sender = get_subgraph_attribute(subgraph_attribute_list_sender, subgraph_id_sender, ent2id_sender)

    # obtain attributes of the receiver's target subgraph
    ent2id_receiver = get_ent2id(file_ent2id_receiver)
    subgraph_attribute_list_receiver = get_subgraph_attribute_list(file_subgraph_attribute_receiver)
    attribute_receiver = get_subgraph_attribute(subgraph_attribute_list_receiver, subgraph_id_receiver, ent2id_receiver)

    '''setup()'''
    pk, mk, s, enc_attributes_sender, enc_attributes_receiver = apabe.setup(attribute_sender, attribute_receiver)

    # use a simple method to generate policy
    attribute_policy = list(set(enc_attributes_sender) & set(enc_attributes_receiver))
    policy = ' and '.join(attribute_policy)

    '''keyGen()'''
    receiver_key = apabe.keygen(pk, mk, policy)

    # get message for encryption
    subgraph_list_sender = get_subgraph_list(file_subgraph_sender)
    msg_sender = pickle.dumps(subgraph_list_sender[subgraph_id_sender]).hex()

    '''encrypt()'''
    if debug:
        print("Encrypt under these attributes: ", enc_attributes_sender)
    ciphertext = apabe.encrypt(pk, s, msg_sender, enc_attributes_sender)

    if debug:
        print(ciphertext)

    '''decrypt()'''
    if len(attribute_policy) >= int(policy_threshold * len(enc_attributes_receiver)):
        rec_msg = apabe.decrypt(ciphertext, receiver_key)
        decrypt_msg = pickle.loads(bytes.fromhex(rec_msg))
        assert decrypt_msg
        if debug:
            print("rec_msg=%s" % str(decrypt_msg))
        assert msg_sender == decrypt_msg
        if debug:
            print("Successful Decryption!")
    else:
        print("Failed Decryption!")


if __name__ == '__main__':
    ''' assume that FB13/FB15K is the sender, FB15K237 is the receiver '''
    sender_file_path = './datasets/FB13/'  # choices = ['FB13', 'FB15K', ...]
    receiver_file_path = './datasets/FB15K237/'

    target_subgraph_id_sender = 1  # specify the target sharing subgraph (multiple subgraphs are also permitted)
    target_subgraph_id_receiver = 12

    kp_abe(sender_file_path, receiver_file_path, target_subgraph_id_sender, target_subgraph_id_receiver)

