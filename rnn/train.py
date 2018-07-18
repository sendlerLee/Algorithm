#!/usr/bin/env python

from __future__ import print_function

import argparse
import os
import random
import sys
import shutil
import time


import numpy as np
import tensorflow as tf
from tensorflow.python.ops import lookup_ops

import vocab_utils
import iterator_utils
import model_helper
import model as rnn_model

min_tf_version = "1.4.0-dev20171024"
if tf.__version__ < min_tf_version :
    raise EnvironmentError("Tensorflow version must >= %s" % min_tf_version)

def add_arguments(parser) :
    parser.register("type", "bool", lambda v : v.lower == "true")

    parser.add_argument("--num_units",type=int,default=32,help="Network Size.")   
    parser.add_argument("--num_layers",type=int,default=2,help="Network depth.")
    parser.add_argument("--time_major",type="bool",default=True,nargs="?",const=True,
            help="Whether to use time-major mode for dynamic RNN.")
    parser.add_argument("--encoder_type", type=str, default="uni", help="""\
      uni | bi | gnmt.
      For bi, we build num_encoder_layers/2 bi-directional layers.
      For gnmt, we build 1 bi-directional layer, and (num_encoder_layers - 1)
        uni-directional layers.\
      """)

    parser.add_argument("--init_op", type=str, default="uniform",
                      help="uniform | glorot_normal | glorot_uniform")
    parser.add_argument("--init_weight", type=float, default=0.1,
                      help=("for uniform init_op, initialize weights "
                            "between [-this, this]."))
    
    parser.add_argument("--optimizer",type=str, default="sgd", help="sgd | adam")
    parser.add_argument("--learning_rate",type=float,default=0.001,
            help="Learning rate. Adam:0.001 | 0.0001")

    parser.add_argument("--num_train_steps", type=int, default=10000, help="Num steps to train.")
    parser.add_argument("--src_max_len",type=int,default=100, help="Max length of src sequences during training.")

    parser.add_argument("--unit_type", type=str, default="lstm",
                      help="lstm | gru | layer_norm_lstm | nas")
    parser.add_argument("--forget_bias", type=float, default=1.0,
                      help="Forget bias for BasicLSTMCell.")
    parser.add_argument("--dropout", type=float, default=0.2,
                      help="Dropout rate (not keep_prob)")
    parser.add_argument("--max_gradient_norm", type=float, default=5.0,
                      help="Clip gradients to this norm.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")

    parser.add_argument("--steps_per_stats", type=int, default=100,
                      help=("How many training steps to do per stats logging."
                            "Save checkpoint every 10x steps_per_stats"))

    parser.add_argument("--max_train", type=int, default=0,
                      help="Limit on the size of training data (0: no limit).")
    parser.add_argument("--num_buckets", type=int, default=5,
                      help="Put data into similar-length buckets.")


    parser.add_argument("--num_keep_ckpts", type=int, default=5,
                      help="Max number of checkpoints to keep.")

    parser.add_argument("--train", type=str, default=None,
                      help="train data.")
    parser.add_argument("--test", type=str, default=None,
                      help="Test data.")
    parser.add_argument("--vocab_file", type=str, default=None,
                      help="vocab dict data.")
    parser.add_argument("--out_dir", type=str, default=None,
                      help="Store log/model files.")
    parser.add_argument("--sos", type=str, default="<s>",
                      help="Start-of-sentence symbol.")
    parser.add_argument("--eos", type=str, default="</s>",
                      help="End-of-sentence symbol.")

    parser.add_argument("--random_seed", type=int, default=None,
                      help="Random seed (>0, set a specific seed).")

    parser.add_argument(
      "--decay_scheme", type=str, default="", help="""\
      How we decay learning rate. Options include:
        luong234: after 2/3 num train steps, we start halving the learning rate
          for 4 times before finishing.
        luong5: after 1/2 num train steps, we start halving the learning rate
          for 5 times before finishing.\
        luong10: after 1/2 num train steps, we start halving the learning rate
          for 10 times before finishing.\
      """)
    

def print_hparams(hparams, skip_patterns=None, header=None):
  """Print hparams, can skip keys based on pattern."""
  if header: print_out("%s" % header)
  values = hparams.values()
  for key in sorted(values.keys()):
    if not skip_patterns or all(
        [skip_pattern not in key for skip_pattern in skip_patterns]):
      print("  %s=%s" % (key, str(values[key])))

    
def create_train_model(hparams) :
    train_file = hparams.train
    vocab_size,vocab_file = vocab_utils.check_vocab(hparams.vocab_file,
              hparams.out_dir,
              sos=hparams.sos,
              eos=hparams.eos,
              unk=vocab_utils.UNK)
    hparams.add_hparam("vocab_size", vocab_size)

    graph = tf.Graph()
    with graph.as_default(),tf.container("train") :
        vocab_table = lookup_ops.index_table_from_file(vocab_file,
                default_value=0)

        iterator = iterator_utils.get_iterator(
            train_file,
            vocab_table,
            batch_size=hparams.batch_size,
            sos=hparams.sos,
            eos=hparams.eos,
            src_max_len=hparams.src_max_len)
        
        model = rnn_model.Model(hparams,
                            mode=tf.contrib.learn.ModeKeys.TRAIN,
                            iterator=iterator,
                            vocab_table=vocab_table)
    return graph,model,iterator
    
def train_fn(hparams) :
    graph,model,iterator = create_train_model(hparams) 
    sess = tf.Session(graph=graph)
    with graph.as_default() :
        loaded_train_model,global_step = model_helper.create_or_load_model(model
                        ,hparams.out_dir,sess,"train") 

    sess.run(model.iterator.initializer)
    while global_step < hparams.num_train_steps :
        start_time = time.time()
        _,train_loss,train_accuracy,train_summary,global_step,_,_,_ = loaded_train_model.train(sess) 
        print (train_loss,train_accuracy)
                        

def main(unused_argv) :
    hparams = tf.contrib.training.HParams(
        train=FLAGS.train,
        test=FLAGS.test,
        vocab_file=FLAGS.vocab_file,
        out_dir=FLAGS.out_dir,

        num_units=FLAGS.num_units,
        num_layers=FLAGS.num_layers,
        time_major=FLAGS.time_major,
        encoder_type=FLAGS.encoder_type,

        init_op=FLAGS.init_op,
        init_weight=FLAGS.init_weight,
        random_seed=FLAGS.random_seed,

        optimizer=FLAGS.optimizer,
        learning_rate=FLAGS.learning_rate,

        num_train_steps=FLAGS.num_train_steps,
        src_max_len=FLAGS.src_max_len,

        unit_type=FLAGS.unit_type,             
        forget_bias=FLAGS.forget_bias,
        dropout=FLAGS.dropout,
        max_gradient_norm=FLAGS.max_gradient_norm,
        batch_size=FLAGS.batch_size,
        steps_per_stats=FLAGS.steps_per_stats,
        
        max_train=FLAGS.max_train,
        num_buckets=FLAGS.num_buckets,
        num_keep_ckpts=FLAGS.num_keep_ckpts,

        decay_scheme=FLAGS.decay_scheme,

        sos=FLAGS.sos,
        eos=FLAGS.eos
    ) 

    print_hparams(hparams)
    out_dir = hparams.out_dir
    shutil.rmtree(out_dir,ignore_errors=True)

    if not tf.gfile.Exists(out_dir) : tf.gfile.MakeDirs(out_dir)

    train_fn(hparams) 


if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    
    FLAGS,unparsed = parser.parse_known_args()
    tf.app.run(main=main,argv=[sys.argv[0]] + unparsed) 
