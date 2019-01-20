#!/usr/bin/env python
# coding: utf-8

# In[19]:


import numpy as np
import pandas as pd
import tensorflow as tf
import datetime
from sklearn.model_selection import train_test_split
import _pickle as pickle
from typing import List
import os
from collections import defaultdict
from tqdm import tqdm
from keras.preprocessing.sequence import pad_sequences
import chakin


# In[20]:


print("Searching for avaiable package.")
chakin.search(lang="English")
DOWNLOAD = bool(input("Download embedding? >>> ").upper() == "Y")
if DOWNLOAD:
    emb_idx = int(input("Index of embedding to download >>> "))
    save_dir = input("Directory to save embeddding ")
    chakin.download(number=emb_idx, save_dir="../data/")


# In[21]:


from data_import import load_embedding_from_disks


# In[22]:


# Parameter
# GLOVE_FILENAME = "../data/glove.840B.300d.txt"
GLOVE_FILENAME = "../data/glove.6B.50d.txt"


# In[23]:


df = pd.read_csv("./text_emotion.csv")
df.head()
SENTI_LST = list(set(df["sentiment"]))
print(SENTI_LST)


# In[24]:


print("Loading embedding from disks...")
word_to_index, index_to_embedding = load_embedding_from_disks(GLOVE_FILENAME, with_indexes=True)
print("Embedding loaded from disks.")


# In[25]:


vocab_size, embedding_dim = index_to_embedding.shape
print(f"Vocab Size: {vocab_size}\nEmbedding Dim: {embedding_dim}")

lstm_units = (512, 1024)
num_classes = 13
iterations = 50
max_seq_length = 40


# In[26]:


def word2int(w: str) -> int:
    try:
        idx = word_to_index[w]
    except KeyError:
        idx = word_to_index["unk"]
    return idx


# In[27]:


X_lst, y_lst = [], []
for sentence, senti in zip(df["content"], df["sentiment"]):
    # ==== Encode x ====
    tokens = sentence.lower().split()
    word_ints = np.array([word2int(x) for x in tokens])
    X_lst.append(word_ints)
    
    # ==== Encode y ====
    label = np.zeros([num_classes])
    senti_index = SENTI_LST.index(senti)
    label[senti_index] = 1
    y_lst.append(label)
    
X_lst = pad_sequences(
    X_lst,
    maxlen=max_seq_length,
    padding="post",
    truncating="post"
)

X_raw = np.stack(X_lst)
y_raw = np.stack(y_lst)
print(X_raw.shape)
print(y_raw.shape)


# In[28]:


(X_train, X_test,
 y_train, y_test) = train_test_split(
    X_raw, y_raw,
    test_size=0.2,
    shuffle=True
)

(X_train, X_val,
 y_train, y_val) = train_test_split(
    X_train, y_train,
    test_size=0.2,
    shuffle=True
)


# In[29]:


print(f"Training and testing set generated,\nX_train shape: {X_train.shape}\ny_train shape: {y_train.shape}\nX_test shape: {X_test.shape}\ny_test shape: {y_test.shape}\nX_validation shape: {X_val.shape}\ny_validation shape: {y_val.shape}")


# In[30]:


X_train_batches = X_train.reshape(25, 1024, max_seq_length)
y_train_batches = y_train.reshape(25, 1024, num_classes)


# In[31]:


try:
    sess.close()
except NameError:
    print("Session already cleaned.")


# In[32]:


tf.reset_default_graph()
sess = tf.InteractiveSession()

with tf.name_scope("DATA_IO"):
    word_ids = tf.placeholder(
        tf.int32,
        shape=[None, max_seq_length]
    )
    
    y = tf.placeholder(
        tf.float32,
        shape=[None, num_classes]
    )

with tf.name_scope("EMBEDDING"):
    embedding = tf.Variable(
        tf.constant(0.0, shape=index_to_embedding.shape),
        trainable=False,
        name="EMBEDDING"
    )
    
    word_representation_layer = tf.nn.embedding_lookup(
        params=embedding,
        ids=word_ids
    )
    
    embedding_placeholder = tf.placeholder(
        tf.float32,
        shape=index_to_embedding.shape
    )
    
    embedding_init = embedding.assign(embedding_placeholder)
    
    _ = sess.run(
        embedding_init, 
            feed_dict={
                embedding_placeholder: index_to_embedding
        }
    )

with tf.name_scope("RNN"):
    cell = tf.nn.rnn_cell.MultiRNNCell(
        [tf.nn.rnn_cell.LSTMCell(
            num_units=units,
            name=f"LSTM_LAYER_{i}")
            for i, units in enumerate(lstm_units)
         ])
    
    lstm_cell = tf.contrib.rnn.DropoutWrapper(
        cell=cell,
        output_keep_prob=0.75
    )
    outputs, state = tf.nn.dynamic_rnn(
        lstm_cell, 
        word_representation_layer,
        dtype=tf.float32
    )

with tf.name_scope("OUTPUT"):
    weight = tf.Variable(
        tf.truncated_normal(
            [lstm_units[-1], num_classes]
        )
    )
    
    bias = tf.Variable(
        tf.random_normal(shape=[num_classes])
    )

# Option i)
    value = tf.transpose(outputs, [1, 0, 2])
    last = tf.gather(value, int(value.get_shape()[0]) - 1)
# Option ii)
#     last = outputs[:, -1, :]
    pred = tf.matmul(last, weight) + bias
    pred_idx = tf.argmax(pred, axis=1)


# In[34]:


with tf.name_scope("METRICS"):
    correct_pred = tf.equal(
        tf.argmax(pred, axis=1),
        tf.argmax(y, axis=1)
    )

    accuracy = tf.reduce_mean(
        tf.cast(correct_pred, tf.float32)
    )

with tf.name_scope("LOSSES"):
    loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=pred,
            labels=y
        )
    )
    optimizer = tf.train.AdamOptimizer().minimize(loss)

tf.summary.scalar("Loss", loss)
tf.summary.scalar("Accuracy", accuracy)


# In[35]:


saver = tf.train.Saver()

sess.run(tf.global_variables_initializer())

merged = tf.summary.merge_all()
logdir = "./tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
writer = tf.summary.FileWriter(logdir, sess.graph)

for e in range(iterations):
    for X_batch, y_batch in zip(X_train_batches, y_train_batches):
        sess.run(
            optimizer,
            feed_dict={
                word_ids: X_batch,
                y: y_batch
            }
        )

    if e % 5 == 0:
        summary = sess.run(
            merged,
            feed_dict={
                word_ids: X_val,
                y: y_val
            }
        )
    if e % 1 == 0:
        train_acc = []
        for X_batch, y_batch in zip(X_train_batches, y_train_batches):
            train_acc.append(accuracy.eval(
                feed_dict={word_ids: X_batch, y: y_batch}
            ))
        avg_tarin_acc = np.mean(train_acc)
        val_acc = accuracy.eval(feed_dict={word_ids: X_val, y: y_val})
        print(
            f"Epochs[{e}]: train batch avg accuracy={avg_tarin_acc}, val accuracy={val_acc}")
    writer.add_summary(summary, e)
    writer.close()
f = lambda src: pred_idx.eval(feed_dict={word_ids: src})
# train_pred = f(X_train)
test_pred = f(X_test)
val_pred = f(X_val)


# In[38]:


def sentence2ints(sentence):
    tokens = sentence.split()
    ids = [word_to_index[word] for word in tokens]
    ids = pad_sequences([ids], maxlen=max_seq_length, padding="post", truncating="post")
    return np.array(ids)


# In[40]:





# In[ ]:




