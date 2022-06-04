from constants import relation, MAX_SEN_LEN
import tensorflow as tf
from transformers import BertTokenizer
import pickle

ADDITIONAL_SPECIAL_TOKENS = ["<e1>", "</e1>", "<e2>", "</e2>"]
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenizer.add_special_tokens({"additional_special_tokens": ADDITIONAL_SPECIAL_TOKENS})

START_E1 = tokenizer.encode('<e1>')[1]
END_E1 = tokenizer.encode('</e1>')[1]
START_E2 = tokenizer.encode('<e2>')[1]
END_E2 = tokenizer.encode('</e2>')[1]


def seperate_file(fi_path, fo1_path, fo2_path, fo3_path, size):
    fi = open(fi_path, 'r')
    fo1 = open(fo1_path, 'w')
    fo2 = open(fo2_path, 'w')
    fo3 = open(fo3_path, 'w')
    for i in range(size):
        l1 = fi.readline()
        l2 = fi.readline()
        l3 = fi.readline()
        l4 = fi.readline()
        l1 = l1.strip().split()
        l1 = l1[1:]
        l1[0] = l1[0][1:]
        l1[len(l1) - 1] = l1[len(l1) - 1][:-1]
        l1 = ' '.join(l1)
        fo1.write(l1)
        fo1.write('\n')
        l1 = l1.replace('<e1>', '')
        l1 = l1.replace('</e1>', '')
        l1 = l1.replace('<e2>', '')
        l1 = l1.replace('</e2>', '')
        fo2.write(l1)
        fo2.write('\n')
        for j in range(len(relation)):
            if relation[j] in l2:
                fo3.write(str(j))
                fo3.write('\n')


def make_token_pickle(filein, fileout):
    fi = open(filein, 'r')
    fo = open(fileout, 'wb')
    x = fi.readlines()
    for s in x:
        token_ids = tokenizer.encode(s)
        pickle.dump(token_ids, fo)


def load_token_pickle(filein, size):
    fi = open(filein, 'rb')
    x = []
    for i in range(size):
        s = pickle.load(fi)
        x.append(s)
    return x


def lengthen_token_seq(a):
    for s in a:
        while len(s) < MAX_SEN_LEN:
            s.append(0)


def make_pos_pickle(filein, fileout):
    fi = open(filein, 'r')
    fo = open(fileout, 'wb')
    x = fi.readlines()
    x_pos = []
    for s in x:
        token_ids = tokenizer.encode(s)
        pos = []
        for i in range(len(token_ids)):
            if token_ids[i] == START_E1:
                pos.append(i)
            if token_ids[i] == END_E1:
                pos.append(i)
            if token_ids[i] == START_E2:
                pos.append(i)
            if token_ids[i] == END_E2:
                pos.append(i)
        x_pos.append(pos)
    pickle.dump(x_pos, fo)


def make_head_e1_e2_pickle(a, out0, out1, out2):
    head_mask = []
    en1_mask = []
    en2_mask = []
    fo0 = open(out0, 'wb')
    fo1 = open(out1, 'wb')
    fo2 = open(out2, 'wb')
    for t in a:
        m0 = []
        for i in range(MAX_SEN_LEN):
            m0.append(0.0)
        m0[0] = 1.0
        head_mask.append(m0)
        m1 = []
        for i in range(MAX_SEN_LEN):
            m1.append(0.0)
        for i in range(t[0], t[1] - 1):
            m1[i] = 1 / (t[1] - 1 - t[0])
        en1_mask.append(m1)
        m2 = []
        for i in range(MAX_SEN_LEN):
            m2.append(0.0)
        for i in range(t[2] - 2, t[3] - 3):
            m2[i] = 1 / ((t[3] - 3) - (t[2] - 2))
        en2_mask.append(m2)
    pickle.dump(head_mask, fo0)
    pickle.dump(en1_mask, fo1)
    pickle.dump(en2_mask, fo2)


def mat_mul(hidden_output, e_mask):
    e_mask = tf.expand_dims(e_mask, 1)
    e_mask = tf.cast(e_mask, tf.float32)
    prod = e_mask @ hidden_output
    prod = tf.squeeze(prod, axis=1)
    return prod
