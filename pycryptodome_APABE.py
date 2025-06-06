""" We provide a simplified implementation of AP-ABE by using pycryptodome library """

from Crypto.PublicKey import ECC, RSA
from Crypto.Cipher import AES, PKCS1_OAEP
from Crypto.Util.number import getRandomNBitInteger, bytes_to_long
from Crypto.Hash import SHA256
import hashlib
import pickle


policy_threshold = 0.01


def get_ent2id(file_path):
    ent2id_lst = []
    with open(file_path, 'r') as file:
        file.readline()
        for line in file:
            ent, idx = line.strip().split('\t')
            ent2id_lst.append(ent.lower())
    return ent2id_lst


def get_subgraph_list(file_path):
    # load multiGraph
    with open(file_path, "rb") as file:
        subgraph_list = pickle.load(file)
    return subgraph_list


def get_subgraph_attribute_list(file_path):
    # load attributes of multiGraph
    with open(file_path, "rb") as file:
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


class APABE:
    def __init__(self):
        """ RSA for attributes encryption, AES for data encryption and decryption """
        self.key_size = 2048
        self.private_key, self.public_key = self.generate_keys()

    def generate_keys(self):
        private_key = RSA.generate(self.key_size)
        public_key = private_key.publickey()
        return private_key, public_key

    def encrypt(self, attribute, msg):
        cipher_rsa = PKCS1_OAEP.new(self.public_key)  # Create RSA encryptor

        sender_key = ''
        encrypted_key = []
        for attri in attribute:
            key = hashlib.sha256(attri.encode()).digest()
            sender_key.join(key.hex())
            encrypted_key.append(cipher_rsa.encrypt(key))

        sender_key = hashlib.sha256(sender_key.encode()).digest()

        cipher = AES.new(sender_key, AES.MODE_GCM)  # Create AES encryptor
        ciphertext, tag = cipher.encrypt_and_digest(msg.encode())

        return encrypted_key, cipher.nonce, tag, ciphertext

    def decrypt(self, receiver_key, encrypted_key, nonce, tag, ciphertext, policy):
        cipher_rsa = PKCS1_OAEP.new(self.private_key)  # create RSA decryptor
        decrypted_key = ''
        for item in encrypted_key:
            decrypted_key.join(cipher_rsa.decrypt(item).hex())
        decrypted_key = hashlib.sha256(decrypted_key.encode()).digest()

        # validate access policy
        if len(policy) >= int(policy_threshold * len(encrypted_key)):
            cipher = AES.new(decrypted_key, AES.MODE_GCM, nonce=nonce)
            decrypted_msg = cipher.decrypt_and_verify(ciphertext, tag)
            return decrypted_msg.decode()
        else:
            print(f'decrypted_key={decrypted_key.hex()}, user_key={receiver_key}')
            raise ValueError("Access Denied: User key does not match the policy")

    def generate_receiver_key(self, attribute):
        receiver_key = []
        for index, item in enumerate(attribute):
            receiver_key.append(hashlib.sha256(item.encode()).hexdigest())
        return receiver_key


# use ECC to simulate bilinear pairing
class BilinearPairingSimulator:
    def __init__(self, curve='P-256'):
        """ initialize the EC parameters """
        self.curve = curve
        self.private_key1 = ECC.generate(curve=self.curve)
        self.g1 = self.private_key1.public_key().pointQ

    def encrypt_attribute(self, attribute_lst):
        enc_attribute = []
        s = getRandomNBitInteger(256)
        y = getRandomNBitInteger(256)

        for idx, attri in enumerate(attribute_lst):
            a = bytes_to_long(SHA256.new(attri.encode()).digest())
            pairing = str(self.g1 * (a * s + y)) + str(self.g1)
            enc_attribute.append(hashlib.sha256(pairing.encode()).hexdigest())
        return enc_attribute


def kp_abe(sender_path, receiver_path, subgraph_id_sender, subgraph_id_receiver):
    simulator = BilinearPairingSimulator()
    apabe = APABE()

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

    # get message for encryption
    subgraph_list_sender = get_subgraph_list(file_subgraph_sender)
    msg_sender = pickle.dumps(subgraph_list_sender[subgraph_id_sender]).hex()

    '''setup()'''
    enc_attribute_sender = simulator.encrypt_attribute(attribute_sender)
    enc_attribute_receiver = simulator.encrypt_attribute(attribute_receiver)

    # use a simplified method to generate policy
    policy = list(set(enc_attribute_sender) & set(enc_attribute_receiver))

    '''keyGen()'''
    receiver_key = apabe.generate_receiver_key(policy)

    '''encrypt()'''
    encrypted_key, nonce, tag, ciphertext = apabe.encrypt(enc_attribute_sender, msg_sender)

    '''decrypt()'''
    try:
        decrypted_msg = apabe.decrypt(receiver_key, encrypted_key, nonce, tag, ciphertext, policy)
        if msg_sender == decrypted_msg:
            print("Successful Decryption!")
            print("decrypted_msg=", pickle.loads(bytes.fromhex(decrypted_msg)))
        else:
            print("Failed Decryption!")
    except ValueError as e:
        print(e)


if __name__ == '__main__':
    ''' assume that FB13/FB15K is the sender, FB15K237 is the receiver '''
    sender_file_path = './datasets/FB13/'  # choices = ['FB13', 'FB15K']
    receiver_file_path = './datasets/FB15K237/'

    target_subgraph_id_sender = 1  # specify the target sharing subgraph (multiple subgraphs are also permitted)
    target_subgraph_id_receiver = 12

    kp_abe(sender_file_path, receiver_file_path, target_subgraph_id_sender, target_subgraph_id_receiver)
