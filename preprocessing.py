from utils import *
from constants import *
import tensorflow as tf


def make_train_files():
    seperate_file(train_path, train_x_ref_path, train_x_path, train_y_path, TRAIN_SIZE)
    make_token_pickle(train_x_path, train_pickle_x)
    make_pos_pickle(train_x_ref_path, train_pickle_x_pos)
    f = open(train_pickle_x_pos, 'rb')
    train_x_pos = pickle.load(f)
    make_head_e1_e2_pickle(train_x_pos, train_x_head_path, train_x_e1_path, train_x_e2_path)
    f.close()


def make_test_files():
    seperate_file(test_path, test_x_ref_path, test_x_path,
                  test_y_path, TEST_SIZE)
    make_token_pickle(test_x_path, test_pickle_x)
    make_pos_pickle(test_x_ref_path, test_pickle_x_pos)
    f = open(test_pickle_x_pos, 'rb')
    test_x_pos = pickle.load(f)
    make_head_e1_e2_pickle(test_x_pos, test_x_head_path, test_x_e1_path, test_x_e2_path)
    f.close()


def get_train_x():
    train_x = load_token_pickle(train_pickle_x, TRAIN_SIZE)
    lengthen_token_seq(train_x)
    train_x = tf.constant(train_x)
    return train_x


def get_train_x_mask():
    fi = open(train_x_head_path, 'rb')
    train_x_head_mask = pickle.load(fi)
    train_x_head_mask = tf.constant(train_x_head_mask)
    fi.close()
    fi = open(train_x_e1_path, 'rb')
    train_x_e1_mask = pickle.load(fi)
    fi.close()
    train_x_e1_mask = tf.constant(train_x_e1_mask)
    fi = open(train_x_e2_path, 'rb')
    train_x_e2_mask = pickle.load(fi)
    fi.close()
    train_x_e2_mask = tf.constant(train_x_e2_mask)
    return train_x_head_mask, train_x_e1_mask, train_x_e2_mask


def get_test_x():
    test_x = load_token_pickle(test_pickle_x, TEST_SIZE)
    lengthen_token_seq(test_x)
    test_x = tf.constant(test_x)
    return test_x


def get_test_x_mask():
    fi = open(test_x_head_path, 'rb')
    test_x_head_mask = pickle.load(fi)
    test_x_head_mask = tf.constant(test_x_head_mask)
    fi.close()
    fi = open(test_x_e1_path, 'rb')
    test_x_e1_mask = pickle.load(fi)
    fi.close()
    test_x_e1_mask = tf.constant(test_x_e1_mask)
    fi = open(test_x_e2_path, 'rb')
    test_x_e2_mask = pickle.load(fi)
    fi.close()
    test_x_e2_mask = tf.constant(test_x_e2_mask)
    return test_x_head_mask, test_x_e1_mask, test_x_e2_mask


def get_train_y():
    fi = open(train_y_path, 'r')
    train_y = []
    for l in fi:
        l = int(l.strip())
        train_y.append(l)
    train_y = tf.keras.utils.to_categorical(train_y)
    return train_y




