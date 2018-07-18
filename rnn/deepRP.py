# -*- coding: UTF-8 _*_
from __future__ import print_function
import datetime
from time import time
import os
import socket
import re
import sys
import cbor
import numpy
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
from tensorflow.contrib import layers

buckets = [50, 100, 150, 200, 250, 300, 350, 400, 500, 9999999]

class DataSetBucket(object):
    def __init__(self, file_name):
        self.instances = []
        self.batch_id = 0
        self.batchs = []
        self.total_batch_size = 0
        print ("Load: " + file_name + " " + str(datetime.datetime.now()) + ".......")
        self.instances = cbor.loads(open(file_name, "rb").read())
        print ("Load Finish " + str(datetime.datetime.now()))
        self.buckets = buckets
        self.num = len(self.instances)

    def shuffle(self):
        self.reset()

        def gen_key_list_pair(buckets):
            return [(i, []) for i in buckets]

        def drop_in(instance, buckets):
            for i in buckets:
                if len(instance[4])-2 < i[0]:
                    i[1].append(instance)
                    return

        # sort instances before drop in buckets
        self.instances = sorted(self.instances, key=lambda x: len(x[2])-2)

        # pre-drop
        key_list = gen_key_list_pair(self.buckets)

        # drop instances into buckets
        for i in self.instances:
            drop_in(i, key_list)
        self.instances = []

        self.batchs = []
        total_case_num = 0
        for index in range(0, len(key_list)):

            (x, arr) = key_list[index]
            itr = int(len(arr) / batch_size)
            last_batch_size = len(arr) % batch_size
            # print(str(itr) + " " + str(last_batch_size))
            begin_itr_id = 0
            end_itr_id = batch_size
            for idx in range(0, itr):
                tmp_arr = arr[begin_itr_id:end_itr_id]
                self.batchs.append(tmp_arr)
                begin_itr_id += batch_size
                end_itr_id += batch_size
                total_case_num += batch_size
            if last_batch_size != 0:
                tmp_arr = arr[begin_itr_id:end_itr_id]
                # self.batchs.append(tmp_arr)
                #print("remove last batch in bucket " + str(x) + " has case num: " + str(last_batch_size))

            key_list[index] = []
        self.total_batch_size = len(self.batchs)

        numpy.random.shuffle(self.batchs)

    def next(self):
        if self.batch_id == self.total_batch_size:
            self.batch_id = 0

        batch_id = self.batch_id
        batch_oid_data = [[i[0]] for i in self.batchs[batch_id]]
        batch_labels_data = [i[1] for i in self.batchs[batch_id]]

        batch_stat_id_data = [i[2] for i in self.batchs[batch_id]]
        batch_stat_val_data = [i[3] for i in self.batchs[batch_id]]
        batch_linkid_data = [i[4][:-2] for i in self.batchs[batch_id]]
        batch_pid_data = [i[4][-2:-1] for i in self.batchs[batch_id]]
        batch_did_data = [[i[4][-1]] for i in self.batchs[batch_id]]
        batch_seq_len_data = [len(i[4][:-2]) for i in self.batchs[batch_id]]
        max_seq_len = max(batch_seq_len_data)

        batch_seq_data = []
        for i in self.batchs[batch_id] :
            order_seq = bytes.decode(i[0])
            seq = int(order_seq.split("_")[1])
            batch_seq_data.append([seq])

        for i in range(0, len(self.batchs[batch_id])) :
            batch_linkid_data[i] = batch_linkid_data[i][0:batch_seq_len_data[i]] + \
                [0.] * (max_seq_len - batch_seq_len_data[i])

        self.batch_id += 1
        return batch_oid_data, batch_labels_data, batch_stat_id_data, batch_stat_val_data, \
            batch_linkid_data, batch_seq_len_data, batch_pid_data, batch_did_data, batch_seq_data

    def has_next(self):
        return self.batch_id < self.total_batch_size

    def reset(self):
        self.batch_id = 0

    def clear(self):
        self.instances = []
        self.batch_id = 0
        self.batchs = []
        self.total_batch_size = 0

class DataSetV2(object):
        def __init__(self, file_name=""):
                self.instances = []
                self.start_id = 0
                start = time()
                self.instances = cbor.loads(open(file_name, "rb").read())
                #self.instances = sorted(self.instances, key = lambda x : x[0])
                elapsed = int(time() - start)
                print(str(datetime.datetime.now()) + "\tLoad Data " + file_name + " Finish" + "\t" + str(elapsed))
                self.num = len(self.instances)

        def shuffle(self):
                self.reset()
                numpy.random.shuffle(self.instances)

        def next(self):
                if self.start_id + batch_size > self.num :
                        self.start_id = 0
                begin_id = self.start_id
                end_id = min(self.start_id + batch_size, self.num)

                #inst = [inst, y, statidfeat, statfvaleat, idlist]
                batch_oid_data = [[i[0]] for i in self.instances[begin_id:end_id]]
                batch_labels_data = [i[1] for i in self.instances[begin_id:end_id]]

                batch_stat_id_data = [i[2] for i in self.instances[begin_id:end_id]]
                batch_stat_val_data = [i[3] for i in self.instances[begin_id:end_id]]
                batch_linkid_data = [i[4][:-2] for i in self.instances[begin_id:end_id]]
                batch_pid_data = [i[4][-2:-1] for i in self.instances[begin_id:end_id]]
                batch_did_data = [[i[4][-1]] for i in self.instances[begin_id:end_id]]
                batch_seq_len_data = [len(i[4][:-2]) for i in self.instances[begin_id:end_id]]
                max_seq_len = max(batch_seq_len_data)

                batch_seq_data = []
                for i in self.instances[begin_id:end_id] :
                        order_seq = bytes.decode(i[0])
                        seq = int(order_seq.split("_")[1])
                        batch_seq_data.append([seq])

                """
                print(str(self.start_id) + "\t" + str(end_id) + "\t" + str(self.num))
                for i in range(batch_size) :
                        print("statlen: " + str(len(batch_stat_val_data[i])))
                        print("statval: " + str(batch_stat_val_data[i]))
                        print("query: " + str(batch_query_data[i]))
                        print(" ".join([str(batch_linkid_data[i]) + "\t" + str(len(batch_linkid_data[i])) + "\n\n"]))
                """


                for i in range(0, end_id - self.start_id) :
                        batch_linkid_data[i] = batch_linkid_data[i][0:batch_seq_len_data[i]] + [0.] * (max_seq_len - batch_seq_len_data[i])

                self.start_id = end_id
                return batch_oid_data, batch_labels_data, batch_stat_id_data, batch_stat_val_data, batch_linkid_data, batch_seq_len_data, batch_pid_data, batch_did_data, batch_seq_data

        def has_next(self):
                return self.start_id + batch_size < self.num

        def reset(self):
                self.start_id = 0 

        def clear(self):
                self.num = 0
                self.start_id = 0
                self.instances = []

HOST = socket.gethostname()
print(HOST)
os.system('rm -rf ./my_graph/' + HOST)

epochs = 20

embedding_size = 16
embedding_size = 32
embedding_size = 64
index_embedding_size = 10

wide_learning_rate = 0.1
wide_learning_beta = 0.1
wide_learning_beta = 0.5
deep_learning_rate = 0.001
deep_learning_rate = 0.0003
deep_learning_rate = 0.0001

nLayes = 5
nHidden = 128
nHidden = 256

nDropout = 0.0

batch_size = 256
test_batch_size = 256
batch_size = 512
test_batch_size = 512


vacab_size = 4000000

# RNN
n_hidden = 128   # hidden layer num of features
n_hidden = 256   # hidden layer num of features
clip_norm = 50000

print("wide_learning_rate: " + str(wide_learning_rate) + "\twide_learning_beta: " + str(wide_learning_beta)+ "\nembedding_size(DNN, FM): " + str(embedding_size) +  "\nnLays: " + str(nLayes) + "\tnHidden: " + str(nHidden) + "\tdeep_learning_rate: " + str(deep_learning_rate) + "\tnDropout: " + str(nDropout) + "\tbatch_size: " + str(batch_size) + "\tnHidden: " + str(n_hidden))

# tf Graph input
oid = tf.placeholder(tf.string, [batch_size, 1])
label = tf.placeholder(tf.float32, [batch_size])

statIdFeat = tf.placeholder(tf.int32, [batch_size, None]) 
statValFeat = tf.placeholder(tf.float32, [batch_size, None]) 
linkidFeat = tf.placeholder(tf.int32, [batch_size, None]) 
pidFeat = tf.placeholder(tf.int32, [batch_size, 1]) 
didFeat = tf.placeholder(tf.int32, [batch_size, 1]) 
seqFeat = tf.placeholder(tf.int32, [batch_size, 1]) 
linkseqlens = tf.placeholder(tf.int32, [batch_size])

# A placeholder for indicating each sequence length
lr_div = tf.placeholder("float")
dropout = tf.placeholder("float")
train_phase = tf.placeholder(tf.bool)

loss_sum = tf.Variable(0.0, name="loss_sum")
count = tf.Variable(0.0, name="count")
reset_op = tf.group(loss_sum.assign(0.0), count.assign(0))
mean_loss = tf.div(loss_sum, count)

def batch_norm_layer(x, train_phase, scope_bn):
        bn_train = batch_norm(x, decay=0.9, center=True, scale=True, updates_collections=None, is_training=True, reuse=None, trainable=True, scope=scope_bn)
        bn_inference = batch_norm(x, decay=0.9, center=True, scale=True, updates_collections=None, is_training=False, reuse=True, trainable=True, scope=scope_bn)
        z = tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)
        return z


def DeepFM(train_phase, dropout_prob):
        Bias = tf.get_variable("wide_bias", [1], initializer=tf.zeros_initializer())

        w_embeddings = tf.get_variable("wide_w_embeddings", [vacab_size], initializer=tf.zeros_initializer())
        w_embedding = tf.nn.embedding_lookup(w_embeddings, statIdFeat)
        statRes = tf.multiply(w_embedding, statValFeat)

        w_linkid_embedding = tf.nn.embedding_lookup(w_embeddings, linkidFeat)
        linkid_mask = tf.sequence_mask(linkseqlens, tf.reduce_max(linkseqlens), tf.float32)
        w_pid_embedding = tf.nn.embedding_lookup(w_embeddings, pidFeat)
        w_did_embedding = tf.nn.embedding_lookup(w_embeddings, didFeat)

        #linear_pred = tf.reduce_sum(statRes, 1) + tf.reduce_sum(tf.multiply(w_linkid_embedding, linkid_mask), 1) + tf.squeeze(w_pid_embedding) + tf.squeeze(w_did_embedding) + Bias
        linear_pred = tf.reduce_sum(statRes, 1) + tf.reduce_sum(tf.multiply(w_linkid_embedding, linkid_mask), 1) + tf.squeeze(w_did_embedding) + Bias
        #linear_pred = tf.reduce_sum(statRes, 1) + Bias
        local_prediction = tf.reshape(linear_pred, [-1])

        fm_embeddings = tf.get_variable("wide_fm_embeddings", [vacab_size, embedding_size], initializer=tf.random_uniform_initializer(-0.05, 0.05))
        with tf.variable_scope("deep") :
                did_embedding = tf.squeeze(tf.nn.embedding_lookup(fm_embeddings, didFeat))
                pid_embedding = tf.squeeze(tf.nn.embedding_lookup(fm_embeddings, pidFeat))
                linkid_embedding = tf.nn.embedding_lookup(fm_embeddings, linkidFeat)

                link_idx_embeddings = tf.get_variable("link_idx_embeddings", [20, index_embedding_size], initializer=tf.random_uniform_initializer(-0.05, 0.05))
                link_seq_embedding = tf.squeeze(tf.nn.embedding_lookup(link_idx_embeddings, seqFeat))

                sm_link = tf.sequence_mask(linkseqlens, tf.reduce_max(linkseqlens), tf.float32)
                sm_link_expand = tf.expand_dims(sm_link, -1)

                tempLeftSum = tf.square(tf.reduce_sum(tf.multiply(linkid_embedding, sm_link_expand), 1))
                tempRightSum = tf.reduce_sum(tf.multiply(tf.square(linkid_embedding), sm_link_expand), 1)
                #fm_pred_linear = 0.5 * (tempLeftSum - tempRightSum) / tf.cast(tf.expand_dims(linkseqlens, -1), tf.float32)
                fm_pred_linear = 0.5 * (tempLeftSum - tempRightSum) 

                link_embedding = tf.nn.embedding_lookup(fm_embeddings, linkidFeat)
                link_embedding = tf.concat([link_embedding], axis=2)
                link_embedding = tf.multiply(link_embedding, sm_link_expand)


                head_size = 128
                head_size = n_hidden
                attEmbSize = embedding_size
                latentfactor = tf.concat([link_embedding], axis=1) #[batch_size, link_num+2, embedding]
                latentfactor = tf.reshape(latentfactor, [batch_size, -1, attEmbSize])


                kernel_size = 6
                filter_num = 256
                conv1 = tf.multiply(tf.layers.conv1d(inputs=latentfactor, filters=filter_num, kernel_size=kernel_size, activation=tf.nn.relu, padding="same"), sm_link_expand)
                conv2 = tf.multiply(tf.nn.relu(tf.layers.conv1d(inputs=conv1, filters=filter_num, kernel_size=kernel_size, padding="same") + \
                                                                tf.layers.conv1d(inputs=latentfactor, filters=filter_num, kernel_size=kernel_size, padding="same")), sm_link_expand)
                AttentionV = conv2


                #fm_pred = tf.reduce_sum(AttentionV, 1) / tf.cast(tf.expand_dims(linkseqlens, -1), tf.float32)
                fm_pred = tf.reduce_sum(AttentionV, 1) 
                #fm_pred = tf.reduce_max(AttentionV, 1) + tf.reduce_sum(AttentionV, 1) / tf.cast(tf.expand_dims(linkseqlens, -1), tf.float32)

                statResFeat = tf.ones_like(statValFeat) - statValFeat
                dnninput = tf.concat([did_embedding, fm_pred_linear, fm_pred, statValFeat, statResFeat], axis=1)
                lay0dim = embedding_size*2 + 16*2 + n_hidden
                dnninput = tf.reshape(dnninput, [batch_size, lay0dim])

                i = 0
                lay_out_0 = tf.layers.dense(dnninput, nHidden, activation=tf.nn.relu)
                lay_out_1 = tf.layers.dense(lay_out_0, nHidden) + tf.layers.dense(dnninput, nHidden)
                lay_out_1 = tf.nn.relu(lay_out_1)

                i = 2
                lay_out_2 = tf.layers.dense(lay_out_1, nHidden, activation=tf.nn.relu)
                lay_out_2 = tf.nn.relu(lay_out_2)
                lay_out_3 = tf.layers.dense(lay_out_2, nHidden) + lay_out_1
                lay_out_3 = tf.nn.relu(lay_out_3)

                res = tf.layers.dense(lay_out_3, 1)
                res = tf.reshape(res, [-1])
            
                #local_prediction = tf.add(local_prediction, res)
                local_prediction = res

        hypothesis = tf.sigmoid(local_prediction)

        local_cost = tf.contrib.losses.log_loss(hypothesis, label, epsilon=1e-07, scope=None)
        with tf.control_dependencies([local_cost]) :
                update_op = tf.group(count.assign(tf.add(count, tf.cast(tf.size(label), dtype=tf.float32))), loss_sum.assign(tf.add(loss_sum, local_cost)))

        return oid, label, local_prediction, local_cost, update_op

with tf.name_scope("training"):
        oid_train, label_train, pred, cost, update_op_train = DeepFM(train_phase, dropout)
        weights_var = tf.trainable_variables()
        gradients = tf.gradients(cost, weights_var)
        optimizer = tf.train.AdamOptimizer(learning_rate=deep_learning_rate)
        train_op = optimizer.apply_gradients(zip(gradients, weights_var))


tf.get_variable_scope().reuse_variables()
with tf.name_scope("validation") :
        oid_test, label_test, pred_test, cost_test, update_op_test = DeepFM(train_phase, dropout)


def calTop1Sim100(instlist, gtLabel, predval) :
        od2maxval = {}
        od2maxi = {}
        for i in range(len(gtLabel)) :
                order_seq = bytes.decode(instlist[i][0])
                od = order_seq.split("_")[0]
                od2maxval.setdefault(od, -1000000)
                od2maxi.setdefault(od, -10000)

                if predval[i] > od2maxval[od] :
                        od2maxi[od] = i
                        od2maxval[od] = predval[i]

        predictright = 0
        allod = 0
        for od in od2maxi :
                maxi = od2maxi[od]
                if gtLabel[maxi] == 1 : 
                        predictright += 1
                allod += 1

        return predictright, allod, predictright * 1.0 / allod

def calAUC(gtLabel, predval) :
        poscnt = sum(gtLabel)
        negcnt = len(gtLabel) - poscnt
        belownegcnt = negcnt;

        i2p = {}
        i2l = {}
        for i in range(len(gtLabel)) :
                i2p[i] = predval[i]
                i2l[i] = gtLabel[i]

        #print(str(poscnt) + "\t" + str(negcnt))
        score = 0
        for item in sorted(i2p.items(), key = lambda x : x[1], reverse = True) :
                idx, val = item
                if i2l[idx] == 1 :
                        score += belownegcnt * 1.0 / negcnt;
                else :
                        belownegcnt -= 1;
        auc = score * 1.0 / poscnt;
        return auc

def run_validation():
        test_set = [DataSetBucket(test_filename)]

        instlist = []
        labellist = []
        predlist = []
        reset_op.run()
        for cur_test_set in test_set :
                cur_test_set.shuffle()
                while cur_test_set.has_next():
                        batch_oids, batch_labels_data, batch_stat_id_data, batch_stat_val_data, batch_linkid_data, batch_seq_len_data, batch_pid_data, batch_did_data, batch_seq_data = cur_test_set.next()
                        oid_test1, label_pred1, pred_test1, _,_ = sess.run([oid_test, label_test, pred_test, cost_test, update_op_test], feed_dict={oid: batch_oids, label:batch_labels_data,\
                                                statIdFeat:batch_stat_id_data, statValFeat:batch_stat_val_data,\
                                                linkidFeat:batch_linkid_data, linkseqlens:batch_seq_len_data, pidFeat:batch_pid_data, didFeat:batch_did_data, seqFeat:batch_seq_data,\
                                                dropout:0.0, train_phase:False, lr_div:1.0})

                        instlist.extend(oid_test1)
                        labellist.extend(label_pred1)
                        predlist.extend(pred_test1)

                print(len(labellist))
                cur_test_set.clear()
                local_acc = mean_loss.eval()

        rankright, allod, top1sim100 = calTop1Sim100(instlist, labellist, predlist)
        auc = calAUC(labellist, predlist)

        reset_op.run()
        return local_acc, auc, rankright, allod, top1sim100


init = tf.global_variables_initializer()

pred_day=datetime.datetime.strptime("20180408", '%Y%m%d')
pred_day=datetime.datetime.strptime("20180601", '%Y%m%d')
pred_day=datetime.datetime.strptime("20180525", '%Y%m%d')
pred_day=datetime.datetime.strptime("20180528", '%Y%m%d')
trainingDays=61
trainingDays=2
trainingDays=31
trainingDays=1
trainingDays=15
trainingDays=8
trainingDays=57
file_dir = "/nfs/project/rank/traindata/rankfeat_"
#file_dir = "/nfs/private/rank/traindata_nspl2/"
#train_set_path = [file_dir + "XSubMin_%s.cbor" % (pred_day + datetime.timedelta(days=-i)).strftime('%Y%m%d') for i in range(trainingDays, 0, -1)]
#train_set_path = [file_dir + "MaxSubX_%s.cbor" % (pred_day + datetime.timedelta(days=-i)).strftime('%Y%m%d') for i in range(trainingDays, 0, -1)]
train_set_path = [file_dir + "%s.cbor" % (pred_day + datetime.timedelta(days=-i)).strftime('%Y%m%d') for i in range(trainingDays, 0, -1)]
#train_set_path = ["/nfs/private/rank/testfeat.cbor"]
print(train_set_path)

#test_filename = "/nfs/private/rank/traindata_nspl2/XSubMin_test.cbor"
#test_filename = "/nfs/private/rank/traindata_nspl2/MaxSubX_test.cbor"
test_filename = "/nfs/project/rank/testfeat.cbor"
test_filename = "/nfs/private/rank/testfeat.cbor"

# Launch the graph
configure = tf.ConfigProto() 
#configure.gpu_options.allow_growth=True
configure.gpu_options.per_process_gpu_memory_fraction = 1
configure.log_device_placement=False
"""
#configure.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
"""
with tf.Session(config=configure) as sess:
        sess.run(init)
        for epoch in range(0, epochs):
                start = time()
                reset_op.run()
                #numpy.random.shuffle(train_set_path)
                print("Epoch " + str(epoch))
                for d_train_path in train_set_path:
                        if not os.path.isfile(d_train_path) : continue
                        #data_set = DataSetV2(d_train_path)
                        data_set = DataSetBucket(d_train_path)
                        data_set.shuffle()

                        while data_set.has_next() :
                                batch_oids, batch_labels_data, batch_stat_id_data, batch_stat_val_data, batch_linkid_data, batch_seq_len_data, batch_pid_data, batch_did_data, batch_seq_data = data_set.next()
                                _, _ = sess.run([train_op, update_op_train], feed_dict={oid: batch_oids, label:batch_labels_data, statIdFeat:batch_stat_id_data, statValFeat:batch_stat_val_data, \
                                                        linkidFeat:batch_linkid_data, linkseqlens:batch_seq_len_data, pidFeat:batch_pid_data, didFeat:batch_did_data, seqFeat:batch_seq_data,\
                                                        #dropout: nDropout, train_phase: True, lr_div: 1})
                                                        dropout: nDropout, train_phase: True, lr_div: (float(epoch)+1)})

                        data_set.clear()
                        train_loss = mean_loss.eval()

                        test_loss, auc, rankright, allod, top1sim100 = run_validation()
                        elapsed = int(time() - start)
                        top1sim100str = str(round(top1sim100, 4)) + "(" + str(rankright) + "," + str(allod) + ")"
                        print("Epoch " + " ".join([str(x) for x in [epoch, elapsed, train_loss, test_loss, round(auc, 4), top1sim100str]]) + "\n")

                #test_loss, auc, rankright, allod, top1sim100 = run_validation()
                #elapsed = int(time() - start)
                #top1sim100str = str(round(top1sim100, 4)) + "(" + str(rankright) + "," + str(allod) + ")"
                #print("Epoch " + " ".join([str(x) for x in [epoch, elapsed, train_loss, test_loss, round(auc, 4), top1sim100str]]) + "\n\n")
                print("Epoch " + str(epoch) + " Finish~~~\n\n")

        print("Optimization Finished!")