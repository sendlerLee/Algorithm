# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""For loading data into NMT models."""
from __future__ import print_function

import collections

import tensorflow as tf
from tensorflow.python.ops import lookup_ops

data_file="./rp_data/train.txt"
vocab_file="./rp_data/vocab_file"
vocab_table = lookup_ops.index_table_from_file(vocab_file,
        default_value=-1)
batch_size = 12
sos="<s>"
eos="</s>"

graph = tf.get_default_graph()
sess = tf.Session(graph=graph)
output_buffer_size=None
src_max_len=200
if not output_buffer_size:
    output_buffer_size = batch_size * 1000
eos_id = tf.cast(vocab_table.lookup(tf.constant(eos)), tf.int32)
sos_id = tf.cast(vocab_table.lookup(tf.constant(sos)), tf.int32)

def _parse_line(train_data) :
    line = tf.string_split([train_data]).values
    features = line[2:]
    label = tf.string_to_number(line[1],out_type=tf.int32)
    label = tf.cond(tf.equal(label,tf.constant(0)),lambda : tf.constant([1,0]),lambda : tf.constant([0,1]))
    return (features,label)

dataset = tf.data.TextLineDataset(data_file)
dataset = dataset.shuffle(output_buffer_size)
dataset = dataset.map(_parse_line)
if src_max_len :
    dataset = dataset.map(
        lambda features,label :
        (features[:src_max_len],label)).prefetch(output_buffer_size)

dataset = dataset.map(
    lambda features,label : (tf.cast(vocab_table.lookup(features), tf.int32),
                       tf.cast(label,tf.int32))).prefetch(output_buffer_size)


dataset = dataset.map(
    lambda src,tgt :(
    src, tgt, tf.size(src), tf.size(tgt))).prefetch(output_buffer_size)

def batching_func(x):
    return x.padded_batch(
        batch_size,
        padded_shapes=(
            tf.TensorShape([None]),
            tf.TensorShape([None]),
            tf.TensorShape([]),
            tf.TensorShape([])),
        padding_values=(
            sos_id,
            sos_id,
            0,
            0))

batched_dataset = batching_func(dataset)
batched_iter = batched_dataset.make_initializable_iterator()
#(src_ids, tgt_ids, src_seq_len,tgt_seq_len) = (batched_iter.get_next())
line = batched_iter.get_next()

sess.run(tf.global_variables_initializer())
sess.run(tf.tables_initializer())
sess.run(batched_iter.initializer)
#src,tgt,src_len,tgt_len = sess.run([src_ids, tgt_ids, src_seq_len,tgt_seq_len])
#print (src)
#print (tgt)
for i in range(5) :
    print (sess.run(line))

