import numpy as np
import pandas as pd
import tensorflow as tf
import keras


num_dimensions = 300

batch_size = 24
lstm_units = 64
num_classes = 13
iterations = 100000
max_seq_length = 40

tf.reset_default_graph()


with tf.name_scope("DATA_IO"):
    input_data = tf.placeholder(
        tf.int32,
        [batch_size, max_seq_length]
    )
    labels = tf.placeholder(
        tf.float32,
        [batch_size, num_classes]
    )

with tf.name_scope("DATA_PROC"):
    data = tf.Variable(tf.zeros(
        [batch_size, max_seq_length, num_dimensions]
    ))

    tf.nn.embedding_lookup(word_vector, input_data)

with tf.name_scope("RNN"):
    lstm_cell = tf.rnn.BasicLSTMCell(lstm_units)
    lstm_cell = tf.contrib.rnn.DropoutWrapper(
        cell=lstm_cell,
        output_keep_prob=0.75
    )
    value, _ = tf.nn.dynamic_rnn(lstm_cell, data, dtype=tf.float32)
