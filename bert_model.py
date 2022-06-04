from utils import mat_mul
from preprocessing import get_train_x, get_train_x_mask, get_train_y, get_test_x, get_test_x_mask
import tensorflow as tf
from constants import *
import numpy as np
from simple_gmlp import gMLPLayer
import os


class BertModel:
    def __init__(self, encoder, depth):
        self.encoder = encoder
        self.depth = depth
        self.input_ids = None
        self.head_mask = None
        self.e1_mask = None
        self.e2_mask = None

    def _add_inputs(self):
        self.input_ids = tf.keras.layers.Input(shape=(MAX_SEN_LEN,), dtype='int32')
        self.head_mask = tf.keras.layers.Input(shape=(MAX_SEN_LEN,))
        self.e1_mask = tf.keras.layers.Input(shape=(MAX_SEN_LEN,))
        self.e2_mask = tf.keras.layers.Input(shape=(MAX_SEN_LEN,))

    def _bert_layer(self):
        self.bertoutput = self.encoder(self.input_ids)
        emb = self.bertoutput[0]

        x = gMLPLayer(dropout_rate=0.5)(emb)
        for _ in range(self.depth - 1):
            x = gMLPLayer(dropout_rate=0.5)(x)

        cls = mat_mul(x, self.head_mask)
        cls = tf.keras.layers.Dropout(DROPOUT)(cls)
        cls = tf.keras.layers.Dense(EMB_SIZE, activation='tanh')(cls)
        # cls = tf.keras.layers.Dense(EMB_SIZE, activation='relu')(cls)

        e1 = mat_mul(x, self.e1_mask)
        e1 = tf.keras.layers.Dropout(DROPOUT)(e1)
        e1 = tf.keras.layers.Dense(EMB_SIZE, activation='tanh')(e1)
        # e1 = tf.keras.layers.Dense(EMB_SIZE, activation='relu')(e1)

        e2 = mat_mul(x, self.e2_mask)
        e2 = tf.keras.layers.Dropout(DROPOUT)(e2)
        e2 = tf.keras.layers.Dense(EMB_SIZE, activation='tanh')(e2)
        # e2 = tf.keras.layers.Dense(EMB_SIZE, activation='relu')(e2)

        com = tf.keras.layers.concatenate([cls, e1, e2])
        # com = tf.keras.layers.concatenate([e1, e2])

        out = tf.keras.layers.Dropout(DROPOUT)(com)
        # out = tf.keras.layers.Dropout(DROPOUT)(cls)
        out = tf.keras.layers.Dense(len(relation), activation='softmax')(out)
        return out

    def _add_train_ops(self):
        self.model = tf.keras.Model(inputs=[self.input_ids, self.head_mask, self.e1_mask, self.e2_mask],
                                    outputs=self._bert_layer())
        # model = tf.keras.Model(inputs=[input_ids, head_mask], outputs=out)
        # model = tf.keras.Model(inputs=[input_ids, e1_mask, e2_mask], outputs=out)
        self.optimizer = tf.keras.optimizers.Adam(lr=LEARNING_RATE)

    def _train(self):
        if not os.path.exists(TRAINED_MODELS):
            os.makedirs(TRAINED_MODELS)

        self.model.compile(optimizer=self.optimizer, loss='binary_crossentropy', metrics='accuracy')
        print(self.model.summary())
        train_x = get_train_x()
        train_x_head_mask, train_x_e1_mask, train_x_e2_mask = get_train_x_mask()
        train_y = get_train_y()
        self.model.fit([train_x, train_x_head_mask, train_x_e1_mask, train_x_e2_mask], train_y,
                       batch_size=BATCH_SIZE, epochs=NUM_EPOCH)
        self.model.save_weights(TRAINED_MODELS)

    def plot_model(self):
        tf.keras.utils.plot_model(self.model, to_file='model.png', show_shapes=False, show_layer_names=True, rankdir='TB',
                                  expand_nested=False, dpi=300)

    def build(self):
        with tf.device('/device:GPU:0'):
            self._add_inputs()
            self._add_train_ops()
            self._train()
            self.plot_model()

    def predict(self):
        self.model.load_weights(TRAINED_MODELS)
        test_x = get_test_x()
        test_x_head_mask, test_x_e1_mask, test_x_e2_mask = get_test_x_mask()
        pred = self.model.predict([test_x, test_x_head_mask, test_x_e1_mask, test_x_e2_mask])
        for j in range(2):
            fo = open(SCORE + 'proposed_answers.txt', 'w')
            for i in range(len(pred)):
                fo.write(str(8001 + i))
                fo.write('\t')
                fo.write(relation[np.argmax(pred[i])])
                fo.write('\n')
