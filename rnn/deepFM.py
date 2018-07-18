#  Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
# #  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""deepFM Estimator , built with tf.layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import ast
import shutil

import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm

parser = argparse.ArgumentParser()
parser.register("type", "bool", lambda v : v.lower == "true")
# Basic model parameters.
parser.add_argument(
    '--batch_size',
    type=int,
    default=1024,
    help='Number of examples to process in a batch')

parser.add_argument(
    '--data_dir',
    type=str,
    default='/nfs/private/rp_data',
    help='Path to directory containing the dataset')

parser.add_argument(
    '--model_dir',
    type=str,
    default='/tmp/deepfm_model',
    help='The directory where the model will be stored.')

parser.add_argument(
    '--train_epochs', type=int, default=100, help='Number of epochs to train.')

parser.add_argument(
    '--epochs_per_eval', type=int, default=10,
    help='The number of training epochs to run between evaluations.')

parser.add_argument(
    '--batch_norm', type="bool", default=False,nargs="?",const=False,
    help='if use batch_norm deal with data.')

parser.add_argument(
    '--deep_layers', type=str, default="[10,10]",
    help='Number of hidden layers along with number of per layer.')

parser.add_argument(
    '--dropout_deep', type=str, default="[0.5,0.5,0.5]",
    help='Number of hidden layers along with number of per layer.')

parser.add_argument(
    '--dropout_fm', type=str, default="[1,1,1]",
    help='Number of hidden layers along with number of per layer.')

parser.add_argument(
    '--embedding_size', type=int, default=8,
    help='the number of latent factors.')

parser.add_argument(
    '--feature_cnt', type=int, default=47,
    help='the number of total feature.')

parser.add_argument(
    '--random_seed', type=int, default=2018,
    help='the number of random_seed.')

parser.add_argument(
    '--learning_rate', type=float, default=0.001,
    help='the number of random_seed.')

parser.add_argument(
    '--optimizer_type', type=str, default="adam",
    help='the number of random_seed.')

parser.add_argument(
    '--loss_type', type=str, default="logloss",
    help='the number of random_seed.')


def input_fn(data_file,repeat,shuffle,batch_size) :
    
    def parse_feature(feature) :
        word = tf.string_split([feature],":").values
        return tf.string_to_number(word[0],out_type=tf.int32),tf.string_to_number(word[1],out_type=tf.float32)

    def parse_record(raw_record) :
        line = tf.string_split([raw_record]).values
        label = tf.string_to_number(line[0],out_type=tf.float32)
        words = line[1:tf.size(line) - 1]
        indices,values = tf.map_fn(lambda feature : (parse_feature(feature)), words, dtype=(tf.int32, tf.float32))
        return indices,values,label


    dataset = tf.data.TextLineDataset(data_file)
    if shuffle :
    	dataset = dataset.shuffle(buffer_size=10240)
    dataset = dataset.repeat(repeat)
    dataset = dataset.map(parse_record).prefetch(batch_size)

    def batching_func(x):
        return x.padded_batch(
            batch_size,
            padded_shapes=([None],[None],[]),
            padding_values=(tf.cast(0,tf.int32),tf.cast(0,tf.float32),tf.cast(0,tf.float32)))

    batched_dataset = batching_func(dataset)
    iterator = batched_dataset.make_one_shot_iterator()
    indices,values,labels = iterator.get_next()

    return {"feat_index":indices,"feat_value":values,"feat_shape":tf.size(indices[0])},labels

def _initialize_weights(params) : 
    weights = dict()

    # embeddings
    weights["feature_embeddings"] = tf.Variable(tf.random_normal([params.feature_size, params.embedding_size], 0.0, 0.01),
            name="feature_embeddings")  # feature_size * K
    weights["feature_bias"] = tf.Variable(
            tf.random_uniform([params.feature_size, 1], 0.0, 1.0), name="feature_bias")  # feature_size * 1

    # deep layers
    num_layer = len(params.deep_layers)
    input_size = params.feature_size * params.embedding_size
    glorot = np.sqrt(2.0 / (input_size + params.deep_layers[0]))
    weights["layer_0"] = tf.Variable(
        np.random.normal(loc=0, scale=glorot, size=(params.feature_size, params.embedding_size, params.deep_layers[0])), dtype=np.float32)
    weights["bias_0"] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(1, params.deep_layers[0])),
                                                        dtype=np.float32)  # 1 * layers[0]
    for i in range(1, num_layer):
        glorot = np.sqrt(2.0 / (params.deep_layers[i-1] + params.deep_layers[i]))
        weights["layer_%d" % i] = tf.Variable(
            np.random.normal(loc=0, scale=glorot, size=(params.deep_layers[i-1], params.deep_layers[i])),
                dtype=np.float32)  # layers[i-1] * layers[i]
        weights["bias_%d" % i] = tf.Variable(
            np.random.normal(loc=0, scale=glorot, size=(1, params.deep_layers[i])),
                dtype=np.float32)  # 1 * layer[i] 

    # final concat projection layer
    """
    if params.use_fm and params.use_deep:
        input_size = params.feature_size + params.embedding_size + params.deep_layers[-1]
    elif params.use_fm:
        input_size = params.feature_size + params.embedding_size
    elif params.use_deep:
        input_size = params.deep_layers[-1]
    glorot = np.sqrt(2.0 / (input_size + 1))
    weights["concat_projection"] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(input_size, 1)),
                dtype=np.float32)  # layers[i-1]*layers[i]
    weights["concat_bias"] = tf.Variable(tf.constant(0.01), dtype=np.float32)
    """
    return weights

def batch_norm_layer(x, train_phase, scope_bn):
    bn_train = batch_norm(x, decay=0.995, center=True, scale=True, updates_collections=None,
                              is_training=True, reuse=None, trainable=True, scope=scope_bn)
    bn_inference = batch_norm(x, decay=0.995, center=True, scale=True, updates_collections=None,
                                  is_training=False, reuse=True, trainable=True, scope=scope_bn)
    z = tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)
    return z
    
def deepFM(features, labels, params, mode) :
    tf.set_random_seed(params.random_seed)

    weights = _initialize_weights(params)

    # model
    embeddings = tf.nn.embedding_lookup(weights["feature_embeddings"],features["feat_index"])  # None * F * K
    feat_value = tf.reshape(features["feat_value"], shape=[-1, features['feat_shape'], 1])
    embeddings = tf.multiply(embeddings, feat_value)

    # ---------- first order term ----------
    y_first_order = tf.nn.embedding_lookup(weights["feature_bias"], features["feat_index"]) # None * F * 1
    y_first_order = tf.reduce_sum(tf.multiply(y_first_order, feat_value), 2)  # None * F
    y_first_order = tf.nn.dropout(y_first_order, params.dropout_keep_fm[0]) # None * F 

    # ---------- second order term ---------------
    # sum_square part
    summed_features_emb = tf.reduce_sum(embeddings, 1)  # None * K
    summed_features_emb_square = tf.square(summed_features_emb)  # None * K

    # square_sum part
    squared_features_emb = tf.square(embeddings)
    squared_sum_features_emb = tf.reduce_sum(squared_features_emb, 1)  # None * K  

    # second order
    y_second_order = 0.5 * tf.subtract(summed_features_emb_square, squared_sum_features_emb)  # None * K
    y_second_order = tf.nn.dropout(y_second_order, params.dropout_keep_fm[1])  # None * K

    # ---------- Deep component ----------
    #y_deep = tf.sparse_to_dense(features["feat_index"],[params.batch_size,params.feature_size * params.embedding_size],embeddings,0) # None * (F*K)
    y_deep = tf.reshape(embeddings, shape=[-1, features["feat_shape"] * params.embedding_size]) # None * (F*K)
    y_deep = tf.nn.dropout(y_deep, params.dropout_keep_deep[0])
    for i in range(0, len(params.deep_layers)):
        if i == 0 :
            weights_layer_i = tf.nn.embedding_lookup(weights["layer_%d" %i],features["feat_index"]) # None * F * (K * layer[i])
            weights_layer_i = tf.reshape(weights_layer_i, shape=[-1, features["feat_shape"] * params.embedding_size, params.deep_layers[i]]) # None * (F*K) * layer[i]
            # expand_dims None * 1 * (F*K)
            y_deep_expand = tf.expand_dims(y_deep,1)
            y_deep_wx = tf.matmul(y_deep_expand, weights_layer_i)
            y_deep = tf.add(tf.squeeze(y_deep_wx), weights["bias_%d" %i]) # None * layer[i] * 1
        else :
            y_deep = tf.add(tf.matmul(y_deep, weights["layer_%d" %i]), weights["bias_%d" %i]) # None * layer[i] * 1
        if params.batch_norm:
            y_deep = batch_norm_layer(y_deep, train_phase=(mode == tf.estimator.ModeKeys.TRAIN), scope_bn="bn_%d" %i) # None * layer[i] * 1
        y_deep = tf.nn.relu(y_deep)
        y_deep = tf.nn.dropout(y_deep, params.dropout_keep_deep[1+i]) # dropout at each Deep layer 

    # ---------- DeepFM ----------
    if params.use_fm and params.use_deep:
        concat_input = tf.concat([y_first_order, y_second_order, y_deep], axis=1)
    elif self.use_fm:
        concat_input = tf.concat([y_first_order, y_second_order], axis=1)
    elif self.use_deep:
        concat_input = y_deep
    #out = tf.add(tf.matmul(concat_input, weights["concat_projection"]), weights["concat_bias"])
    out = tf.reduce_sum(concat_input,1,keepdims=True)

    return out


def _model_fn(features, labels, mode, params):
    """Model function for DeepFM."""
    preds = deepFM(features, labels, params, mode)
    predictions = {
        'prob': tf.nn.sigmoid(preds,name="tensor_sigmoid")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # loss
    labels = tf.reshape(labels, shape=[tf.size(labels), 1])
    if params.loss_type == "logloss":
        out = tf.nn.sigmoid(preds)
        loss = tf.losses.log_loss(labels, out)
    elif params.loss_type == "mse":
        loss = tf.nn.l2_loss(tf.subtract(labels, preds))
    #loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)
    #loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.reshape(preds,[-1]), labels=tf.reshape(labels,[-1])))

    tf.identity(loss, name='cross_entropy')
    tf.summary.scalar('cross_entropy', loss)

    # Configure the training op
    if mode == tf.estimator.ModeKeys.TRAIN:
        if params.optimizer_type == "adam":
            optimizer = tf.train.AdamOptimizer(learning_rate=params.learning_rate, beta1=0.9, beta2=0.999,
                                                        epsilon=1e-8)
        elif params.optimizer_type == "adagrad":
            optimizer = tf.train.AdagradOptimizer(learning_rate=params.learning_rate,
                                                           initial_accumulator_value=1e-8)
        elif params.optimizer_type == "gd":
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=params.learning_rate)
        elif paparams.optimizer_type == "momentum":
            optimizer = tf.train.MomentumOptimizer(learning_rate=params.learning_rate, momentum=0.95)
        train_op = optimizer.minimize(loss, tf.train.get_or_create_global_step())
    else:
        train_op = None

    auc = tf.metrics.auc( labels , predictions['prob'])
    mse = tf.metrics.mean_squared_error( labels , predictions['prob'])
    #acc = tf.metrics.accuracy(labels , predictions['prob'])
    metrics = {'auc': auc,'mse' : mse}

    # Create a tensor named train_accuracy for logging purposes
    #tf.identity(auc, name='train_auc')
    #tf.summary.scalar('train_auc', auc)


    return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=predictions,
      loss=loss,
      train_op=train_op,
      eval_metric_ops=metrics)


def main(unused_argv):

    model_params = tf.contrib.training.HParams(
      use_fm=True,
      use_deep=True,
      train_epochs=FLAGS.train_epochs,
      batch_size=FLAGS.batch_size,
      batch_norm=FLAGS.batch_norm,
      random_seed=FLAGS.random_seed,
      deep_layers=ast.literal_eval(FLAGS.deep_layers),
      dropout_keep_fm=ast.literal_eval(FLAGS.dropout_fm),
      dropout_keep_deep=ast.literal_eval(FLAGS.dropout_deep),
      epochs_per_eval=FLAGS.epochs_per_eval,
      embedding_size=FLAGS.embedding_size,
      feature_size=FLAGS.feature_cnt,
      optimizer_type=FLAGS.optimizer_type,
      loss_type=FLAGS.loss_type,
      learning_rate=FLAGS.learning_rate,
      model_dir=FLAGS.model_dir,
      data_dir=FLAGS.data_dir)

    # Create the Estimator
    _classifier = tf.estimator.Estimator(
          model_fn=_model_fn,
          model_dir=FLAGS.model_dir,
          params=model_params)

    # Set up training hook that logs the training accuracy every 100 steps.
    tensors_to_log = {'probabilities': 'tensor_sigmoid'}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=100)

    # Train the model
    for n in range(FLAGS.train_epochs // FLAGS.epochs_per_eval):
       	_classifier.train(input_fn=lambda :input_fn(model_params.data_dir + "/train",FLAGS.epochs_per_eval,True, model_params.batch_size))#, hooks=[logging_hook])
        eval_results = _classifier.evaluate(input_fn=lambda :input_fn(model_params.data_dir + "/test",1,False,model_params.batch_size))
        print('Results at epoch ', (n + 1) * FLAGS.epochs_per_eval)
        print('-' * 60)
        print('Evaluation results:\n\t%s' % eval_results)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    FLAGS, unparsed = parser.parse_known_args()
    shutil.rmtree(FLAGS.model_dir, ignore_errors=True)
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
