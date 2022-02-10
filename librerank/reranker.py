import itertools

import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell, static_bidirectional_rnn, LSTMCell, MultiRNNCell
import numpy as np
import heapq


def tau_function(x):
    return tf.where(x > 0, tf.exp(x), tf.zeros_like(x))


def attention_score(x):
    return tau_function(x) / tf.add(tf.reduce_sum(tau_function(x), axis=1, keepdims=True), 1e-20)


class BaseModel(object):
    def __init__(self, feature_size, eb_dim, hidden_size, max_time_len, itm_spar_num, itm_dens_num,
                 profile_num, max_norm=None):
        # reset graph
        tf.reset_default_graph()
        self.graph = tf.Graph()
        with self.graph.as_default():

            # input placeholders
            with tf.name_scope('inputs'):
                self.itm_spar_ph = tf.placeholder(tf.int32, [None, max_time_len, itm_spar_num], name='item_spar')
                self.itm_dens_ph = tf.placeholder(tf.float32, [None, max_time_len, itm_dens_num], name='item_dens')
                self.usr_profile = tf.placeholder(tf.int32, [None, profile_num], name='usr_profile')
                # self.usr_spar_ph = tf.placeholder(tf.int32, [None, max_seq_len, hist_spar_num], name='user_spar')
                # self.usr_dens_ph = tf.placeholder(tf.float32, [None, max_seq_len, hist_dens_num], name='user_dens')
                self.seq_length_ph = tf.placeholder(tf.int32, [None, ], name='seq_length_ph')
                # self.hist_length_ph = tf.placeholder(tf.int32, [None, ], name='hist_length_ph')
                self.label_ph = tf.placeholder(tf.float32, [None, max_time_len], name='label_ph')
                # self.time_ph = tf.placeholder(tf.float32, [None, max_seq_len], name='time_ph')
                self.is_train = tf.placeholder(tf.bool, [], name='is_train')


                # lr
                self.lr = tf.placeholder(tf.float32, [])
                # reg lambda
                self.reg_lambda = tf.placeholder(tf.float32, [])
                # keep prob
                self.keep_prob = tf.placeholder(tf.float32, [])
                self.max_time_len = max_time_len
                # self.max_seq_len = max_seq_len
                self.hidden_size = hidden_size
                self.emb_dim = eb_dim
                self.itm_spar_num = itm_spar_num
                self.itm_dens_num = itm_dens_num
                # self.hist_spar_num = hist_spar_num
                # self.hist_dens_num = hist_dens_num
                self.profile_num = profile_num
                self.max_grad_norm = max_norm
                self.ft_num = itm_spar_num * eb_dim + itm_dens_num
                self.feature_size = feature_size

            # embedding
            with tf.name_scope('embedding'):
                self.emb_mtx = tf.get_variable('emb_mtx', [feature_size + 1, eb_dim],
                                               initializer=tf.truncated_normal_initializer)
                self.itm_spar_emb = tf.gather(self.emb_mtx, self.itm_spar_ph)
                # self.usr_spar_emb = tf.gather(self.emb_mtx, self.usr_spar_ph)
                # self.usr_prof_emb = tf.gather(self.emb_mtx, self.usr_profile)

                self.item_seq = tf.concat(
                    [tf.reshape(self.itm_spar_emb, [-1, max_time_len, itm_spar_num * eb_dim]), self.itm_dens_ph], axis=-1)

    def build_fc_net(self, inp, scope='fc'):
        with tf.variable_scope(scope):
            bn1 = tf.layers.batch_normalization(inputs=inp, name='bn1', training=self.is_train)
            fc1 = tf.layers.dense(bn1, 200, activation=tf.nn.relu, name='fc1')
            dp1 = tf.nn.dropout(fc1, self.keep_prob, name='dp1')
            fc2 = tf.layers.dense(dp1, 80, activation=tf.nn.relu, name='fc2')
            dp2 = tf.nn.dropout(fc2, self.keep_prob, name='dp2')
            fc3 = tf.layers.dense(dp2, 2, activation=None, name='fc3')
            score = tf.nn.softmax(fc3)
            score = tf.reshape(score[:, :, 0], [-1, self.max_time_len])
            # output
            seq_mask = tf.sequence_mask(self.seq_length_ph, maxlen=self.max_time_len, dtype=tf.float32)
            y_pred = seq_mask * score
        return y_pred

    def build_mlp_net(self, inp, layer=(500, 200, 80), scope='mlp'):
        with tf.variable_scope(scope):
            inp = tf.layers.batch_normalization(inputs=inp, name='mlp_bn', training=self.is_train)
            for i, hidden_num in enumerate(layer):
                fc = tf.layers.dense(inp, hidden_num, activation=tf.nn.relu, name='fc' + str(i))
                inp = tf.nn.dropout(fc, self.keep_prob, name='dp' + str(i))
            final = tf.layers.dense(inp, 1, activation=None, name='fc_final')
            score = tf.reshape(final, [-1, self.max_time_len])
            seq_mask = tf.sequence_mask(self.seq_length_ph, maxlen=self.max_time_len, dtype=tf.float32)
            y_pred = seq_mask * score
        return y_pred

    def build_logloss(self, y_pred):
        # loss
        self.loss = tf.losses.log_loss(self.label_ph, y_pred)
        self.opt()

    def build_norm_logloss(self, y_pred):
        self.loss = - tf.reduce_sum(self.label_ph/(tf.reduce_sum(self.label_ph, axis=-1, keepdims=True) + 1e-8) * tf.log(y_pred))
        self.opt()

    def build_mseloss(self, y_pred):
        self.loss = tf.losses.mean_squared_error(self.label_ph, y_pred)
        self.opt()

    def build_attention_loss(self, y_pred):
        self.label_wt = attention_score(self.label_ph)
        self.pred_wt = attention_score(y_pred)
        # self.pred_wt = y_pred
        self.loss = tf.losses.log_loss(self.label_wt, self.pred_wt)
        # self.loss = tf.losses.mean_squared_error(self.label_wt, self.pred_wt)
        self.opt()

    def opt(self):
        for v in tf.trainable_variables():
            if 'bias' not in v.name and 'emb' not in v.name:
                self.loss += self.reg_lambda * tf.nn.l2_loss(v)
                # self.loss += self.reg_lambda * tf.norm(v, ord=1)

        # self.lr = tf.train.exponential_decay(
        #     self.lr_start, self.global_step, self.lr_decay_step,
        #     self.lr_decay_rate, staircase=True, name="learning_rate")


        self.optimizer = tf.train.AdamOptimizer(self.lr)

        if self.max_grad_norm > 0:
            grads_and_vars = self.optimizer.compute_gradients(self.loss)
            for idx, (grad, var) in enumerate(grads_and_vars):
                if grad is not None:
                    grads_and_vars[idx] = (tf.clip_by_norm(grad, self.max_grad_norm), var)
            self.train_step = self.optimizer.apply_gradients(grads_and_vars)
        else:
            self.train_step = self.optimizer.minimize(self.loss)

    def multihead_attention(self,
                            queries,
                            keys,
                            num_units=None,
                            num_heads=2,
                            scope="multihead_attention",
                            reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            if num_units is None:
                num_units = queries.get_shape().as_list()[-1]

            Q = tf.layers.dense(queries, num_units, activation=None)  # (N, T_q, C)
            K = tf.layers.dense(keys, num_units, activation=None)  # (N, T_k, C)
            V = tf.layers.dense(keys, num_units, activation=None)  # (N, T_k, C)

            Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
            K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
            V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)

            outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)
            outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

            key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))  # (N, T_k)
            key_masks = tf.tile(key_masks, [num_heads, 1])  # (h*N, T_k)
            key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])  # (h*N, T_q, T_k)

            paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)  # (h*N, T_q, T_k)
            outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

            query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1)))  # (N, T_q)
            query_masks = tf.tile(query_masks, [num_heads, 1])  # (h*N, T_q)
            query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
            outputs *= query_masks  # broadcasting. (N, T_q, C)
            outputs = tf.nn.dropout(outputs, self.keep_prob)
            outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)
            outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)

        return outputs

    def positionwise_feed_forward(self, inp, d_hid, d_inner_hid, dropout=0.9):
        with tf.variable_scope('pos_ff'):
            inp = tf.layers.batch_normalization(inputs=inp, name='bn1', training=self.is_train)
            l1 = tf.layers.conv1d(inp, d_inner_hid, 1, activation='relu')
            l2 = tf.layers.conv1d(l1, d_hid, 1)
            dp = tf.nn.dropout(l2, dropout, name='dp')
            dp = dp + inp
            output = tf.layers.batch_normalization(inputs=dp, name='bn2', training=self.is_train)
        return output

    def bilstm(self, inp, hidden_size, scope='bilstm', reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias=1.0, name='cell_fw')
            lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias=1.0, name='cell_bw')

            outputs, state_fw, state_bw = static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, inp, dtype='float32')
        return outputs, state_fw, state_bw


    def train(self, batch_data, lr, reg_lambda, keep_prob=0.8):
        with self.graph.as_default():
            loss, _ = self.sess.run([self.loss, self.train_step], feed_dict={
                self.usr_profile: np.reshape(np.array(batch_data[1]), [-1, self.profile_num]),
                self.itm_spar_ph: batch_data[2],
                self.itm_dens_ph: batch_data[3],
                # self.usr_spar_ph: batch_data[3],
                # self.usr_dens_ph: batch_data[4],
                self.label_ph: batch_data[4],
                self.seq_length_ph: batch_data[6],
                # self.hist_length_ph: batch_data[8],
                self.lr: lr,
                self.reg_lambda: reg_lambda,
                self.keep_prob: keep_prob,
                self.is_train: True,
            })
            return loss

    def eval(self, batch_data, reg_lambda, keep_prob=1, no_print=True):
        with self.graph.as_default():
            pred, loss = self.sess.run([self.y_pred, self.loss], feed_dict={
                self.usr_profile: np.reshape(np.array(batch_data[1]), [-1, self.profile_num]),
                self.itm_spar_ph: batch_data[2],
                self.itm_dens_ph: batch_data[3],
                # self.usr_spar_ph: batch_data[3],
                # self.usr_dens_ph: batch_data[4],
                self.label_ph: batch_data[4],
                self.seq_length_ph: batch_data[6],
                # self.hist_length_ph: batch_data[8],
                self.reg_lambda: reg_lambda,
                self.keep_prob: keep_prob,
                self.is_train: False,
            })
            return pred.reshape([-1, self.max_time_len]).tolist(), loss

    def save(self, path):
        with self.graph.as_default():
            saver = tf.train.Saver()
            saver.save(self.sess, save_path=path)
            print('Save model:', path)

    def load(self, path):
        with self.graph.as_default():
            ckpt = tf.train.get_checkpoint_state(path)
            if ckpt and ckpt.model_checkpoint_path:
                saver = tf.train.Saver()
                saver.restore(sess=self.sess, save_path=ckpt.model_checkpoint_path)
                print('Restore model:', ckpt.model_checkpoint_path)

    def set_sess(self, sess):
        self.sess = sess


class GSF(BaseModel):
    def __init__(self, feature_size, eb_dim, hidden_size, max_time_len, itm_spar_num, itm_dens_num,
                 profile_num, max_norm=None, group_size=1, activation='relu', hidden_layer_size=[512, 256, 128]):
        super(GSF, self).__init__(feature_size, eb_dim, hidden_size, max_time_len,
                                  itm_spar_num, itm_dens_num, profile_num, max_norm)

        with self.graph.as_default():
            self.group_size = group_size
            input_list = tf.unstack(self.item_seq, axis=1)
            input_data = tf.concat(input_list, axis=0)
            output_data = input_data
            if activation == 'elu':
                activation = tf.nn.elu
            else:
                activation = tf.nn.relu

            input_data_list = tf.split(output_data, self.max_time_len, axis=0)
            output_sizes = hidden_layer_size + [group_size]
            #
            output_data_list = [0 for _ in range(max_time_len)]
            group_list = []
            self.get_possible_group([], group_list)
            for group in group_list:
                group_input = tf.concat([input_data_list[idx]
                                         for idx in group], axis=1)
                group_score_list = self.build_gsf_fc_function(group_input, output_sizes, activation)
                for i in range(group_size):
                    output_data_list[group[i]] += group_score_list[i]
            self.y_pred = tf.concat(output_data_list, axis=1)
            self.y_pred = tf.nn.softmax(self.y_pred, axis=-1)
            self.build_norm_logloss(self.y_pred)

    def build_gsf_fc_function(self, inp, hidden_size, activation, scope="gsf_nn"):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            for j in range(len(hidden_size)):
                bn = tf.layers.batch_normalization(inputs=inp, name='bn'+str(j), training=self.is_train)
                if j != len(hidden_size) - 1:
                    inp = tf.layers.dense(bn, hidden_size[j], activation=activation, name='fc' + str(j))
                else:
                    inp = tf.layers.dense(bn, hidden_size[j], activation=tf.nn.sigmoid, name='fc' + str(j))
        return tf.split(inp, self.group_size, axis=1)

    def get_possible_group(self, group, group_list):
        if len(group) == self.group_size:
            group_list.append(group)
            return
        else:
            for i in range(self.max_time_len):
                self.get_possible_group(group + [i], group_list)


class miDNN(BaseModel):
    def __init__(self, feature_size, eb_dim, hidden_size, max_time_len, itm_spar_num, itm_dens_num,
                 profile_num, max_norm=None, hidden_layer_size=[512, 256, 128]):
        super(miDNN, self).__init__(feature_size, eb_dim, hidden_size, max_time_len,
                                itm_spar_num, itm_dens_num, profile_num, max_norm)

        with self.graph.as_default():
            fmax = tf.reduce_max(tf.reshape(self.item_seq, [-1, self.max_time_len, self.ft_num]), axis=1, keep_dims=True)
            fmin = tf.reduce_min(tf.reshape(self.item_seq, [-1, self.max_time_len, self.ft_num]), axis=1, keep_dims=True)
            global_seq = (self.item_seq - fmin)/(fmax - fmin + 1e-8)
            inp = tf.concat([self.item_seq, global_seq], axis=-1)

            self.y_pred = self.build_miDNN_net(inp, hidden_layer_size)
            self.build_logloss(self.y_pred)


    def build_miDNN_net(self, inp, layer, scope='mlp'):
        with tf.variable_scope(scope):
            inp = tf.layers.batch_normalization(inputs=inp, name='mlp_bn', training=self.is_train)
            for i, hidden_num in enumerate(layer):
                fc = tf.layers.dense(inp, hidden_num, activation=tf.nn.relu, name='fc' + str(i))
                inp = tf.nn.dropout(fc, self.keep_prob, name='dp' + str(i))
            final = tf.layers.dense(inp, 1, activation=tf.nn.sigmoid, name='fc_final')
            score = tf.reshape(final, [-1, self.max_time_len])
            seq_mask = tf.sequence_mask(self.seq_length_ph, maxlen=self.max_time_len, dtype=tf.float32)
            y_pred = seq_mask * score
        return y_pred


class PRM(BaseModel):
    def __init__(self, feature_size, eb_dim, hidden_size, max_time_len, itm_spar_num, itm_dens_num,
                 profile_num, max_norm=None, d_model=64, d_inner_hid=128, n_head=1):
        super(PRM, self).__init__(feature_size, eb_dim, hidden_size, max_time_len,
                                 itm_spar_num, itm_dens_num, profile_num, max_norm)

        with self.graph.as_default():
            pos_dim = self.item_seq.get_shape().as_list()[-1]
            self.d_model = d_model
            self.pos_mtx = tf.get_variable("pos_mtx", [max_time_len, pos_dim],
                                       initializer=tf.truncated_normal_initializer)
            self.item_seq = self.item_seq + self.pos_mtx
            if pos_dim % 2:
                self.item_seq = tf.pad(self.item_seq, [[0, 0], [0, 0], [0, 1]])

            self.item_seq = self.multihead_attention(self.item_seq, self.item_seq, num_units=d_model, num_heads=n_head)
            self.item_seq = self.positionwise_feed_forward(self.item_seq, self.d_model, d_inner_hid, self.keep_prob)
            # self.item_seq = tf.layers.dense(self.item_seq, self.d_model, activation=tf.nn.tanh, name='fc')

            mask = tf.expand_dims(tf.sequence_mask(self.seq_length_ph, maxlen=max_time_len, dtype=tf.float32), axis=-1)
            seq_rep = self.item_seq * mask

            self.y_pred = self.build_prm_fc_function(seq_rep)
            # self.y_pred = self.build_fc_net(seq_rep)
            self.build_logloss(self.y_pred)

    def build_prm_fc_function(self, inp):
        bn1 = tf.layers.batch_normalization(inputs=inp, name='bn1', training=self.is_train)
        fc1 = tf.layers.dense(bn1, self.d_model, activation=tf.nn.relu, name='fc1')
        dp1 = tf.nn.dropout(fc1, self.keep_prob, name='dp1')
        fc2 = tf.layers.dense(dp1, 1, activation=None, name='fc2')
        score = tf.nn.softmax(tf.reshape(fc2, [-1, self.max_time_len]))
        # output
        seq_mask = tf.sequence_mask(self.seq_length_ph, maxlen=self.max_time_len, dtype=tf.float32)
        return seq_mask * score


class SetRank(BaseModel):
    def __init__(self, feature_size, eb_dim, hidden_size, max_time_len, itm_spar_num, itm_dens_num,
                 profile_num, max_norm=None, d_model=256, n_head=8, d_inner_hid=64):
        super(SetRank, self).__init__(feature_size, eb_dim, hidden_size, max_time_len,
                                 itm_spar_num, itm_dens_num, profile_num, max_norm)

        with self.graph.as_default():
            self.item_seq = self.multihead_attention(self.item_seq, self.item_seq, num_units=d_model, num_heads=n_head)
            self.item_seq = self.positionwise_feed_forward(self.item_seq, d_model, d_inner_hid, dropout=self.keep_prob)

            mask = tf.expand_dims(tf.sequence_mask(self.seq_length_ph, maxlen=max_time_len, dtype=tf.float32), axis=-1)
            seq_rep = self.item_seq * mask

            self.y_pred = self.build_fc_net(seq_rep)
            self.build_attention_loss(self.y_pred)


class DLCM(BaseModel):
    def __init__(self, feature_size, eb_dim, hidden_size, max_time_len, itm_spar_num, itm_dens_num,
                 profile_num, max_norm=None):
        super(DLCM, self).__init__(feature_size, eb_dim, hidden_size, max_time_len,
                                   itm_spar_num, itm_dens_num, profile_num, max_norm)

        with self.graph.as_default():
            with tf.name_scope('gru'):
                seq_ht, seq_final_state = tf.nn.dynamic_rnn(GRUCell(hidden_size), inputs=self.item_seq,
                                                            sequence_length=self.seq_length_ph, dtype=tf.float32,
                                                            scope='gru1')
            self.y_pred = self.build_phi_function(seq_ht, seq_final_state, hidden_size)
            self.build_attention_loss(self.y_pred)
            # self.build_logloss()

    def build_phi_function(self, seq_ht, seq_final_state, hidden_size):
        bn1 = tf.layers.batch_normalization(inputs=seq_final_state, name='bn1', training=self.is_train)
        seq_final_fc = tf.layers.dense(bn1, hidden_size, activation=tf.nn.tanh, name='fc1')
        dp1 = tf.nn.dropout(seq_final_fc, self.keep_prob, name='dp1')
        seq_final_fc = tf.expand_dims(dp1, axis=1)
        bn2 = tf.layers.batch_normalization(inputs=seq_ht, name='bn2', training=self.is_train)
        # fc2 = tf.layers.dense(tf.multiply(bn2, seq_final_fc), 2, activation=None, name='fc2')
        # score = tf.nn.softmax(fc2)
        # score = tf.reshape(score[:, :, 0], [-1, self.max_time_len])
        fc2 = tf.layers.dense(tf.multiply(bn2, seq_final_fc), 1, activation=None, name='fc2')
        score = tf.reshape(fc2, [-1, self.max_time_len])
        # sequence mask
        seq_mask = tf.sequence_mask(self.seq_length_ph, maxlen=self.max_time_len, dtype=tf.float32)
        score = score * seq_mask
        score = score - tf.reduce_min(score, 1, keep_dims=True)
        return score


class EGR_base(BaseModel):
    def __init__(self, feature_size, eb_dim, hidden_size, max_time_len, itm_spar_num, itm_dens_num,
                 profile_num, max_norm=None):
        super(EGR_base, self).__init__(feature_size, eb_dim, hidden_size, max_time_len,
                                      itm_spar_num, itm_dens_num, profile_num, max_norm)
        # global feature
        # new_shop_feature = self.get_global_feature(self.item_seq)
        with self.graph.as_default():
            new_shop_feature = self.item_seq

            with tf.variable_scope("network"):
                layer1 = new_shop_feature
                # fn = tf.nn.relu
                # layer1 = tf.layers.dense(dense_feature_normed, 128, name='layer1', activation=fn)
                new_dense_feature, final_state = tf.nn.dynamic_rnn(GRUCell(hidden_size), inputs=layer1,
                                                            sequence_length=self.seq_length_ph, dtype=tf.float32,
                                                            scope='gru')
                new_feature = tf.concat([new_shop_feature, new_dense_feature], axis=-1)

                self.y_pred = self.build_fc_net(new_feature)

    def get_global_feature(self, inputph):
        tensor_global_max = tf.reduce_max(inputph, axis=1, keep_dims=True)  # (B, 1, d2)
        tensor_global_min = tf.reduce_min(inputph, axis=1, keep_dims=True)  # (B, 1, d2)
        tensor_global_max_tile = tf.tile(tensor_global_max, [1, self.max_time_len, 1])  # (B, 17, d2)
        tensor_global_min_tile = tf.tile(tensor_global_min, [1, self.max_time_len, 1])  # (B, 17, d2)
        matrix_f_global = tf.where(tf.equal(tensor_global_max_tile, tensor_global_min_tile),
                                   tf.fill(tf.shape(inputph), 0.5),
                                   tf.div(tf.subtract(inputph, tensor_global_min_tile),
                                          tf.subtract(tensor_global_max_tile, tensor_global_min_tile)))

        tensor_global_mean = tf.divide(tf.reduce_sum(matrix_f_global, axis=1, keep_dims=True),
                                       tf.cast(self.max_time_len, dtype=tf.float32))  # (B, 1, d2)
        tensor_global_mean_tile = tf.tile(tensor_global_mean, [1, self.max_time_len, 1])  # (B, 17, d2)
        tensor_global_sigma = tf.square(matrix_f_global - tensor_global_mean_tile)  # (B, 1, d2)

        new_shop_feature = tf.concat(
            [inputph, tensor_global_max_tile, tensor_global_min_tile, matrix_f_global, tensor_global_mean_tile,
             tensor_global_sigma], axis=2)
        return new_shop_feature


class EGR_evaluator(EGR_base):
    def __init__(self, feature_size, eb_dim, hidden_size, max_time_len, itm_spar_num, itm_dens_num,
                 profile_num, max_norm=None):
        super(EGR_evaluator, self).__init__(feature_size, eb_dim, hidden_size, max_time_len,
                                      itm_spar_num, itm_dens_num, profile_num, max_norm)
        with self.graph.as_default():
            self.build_logloss(self.y_pred)

    def predict(self, item_spar_fts, item_dens_fts, seq_len):
        with self.graph.as_default():
            ctr_probs = self.sess.run(self.y_pred, feed_dict={
                self.itm_spar_ph: item_spar_fts.reshape([-1, self.max_time_len, self.itm_spar_num]),
                self.itm_dens_ph: item_dens_fts.reshape([-1, self.max_time_len, self.itm_dens_num]),
                self.seq_length_ph: seq_len,
                self.keep_prob: 1.0,
                self.is_train: False})
            return ctr_probs


class EGR_discriminator(EGR_base):
    def __init__(self, feature_size, eb_dim, hidden_size, max_time_len, itm_spar_num, itm_dens_num,
                 profile_num, max_norm=None, c_entropy_d=0.001):
        super(EGR_discriminator, self).__init__(feature_size, eb_dim, hidden_size, max_time_len,
                                      itm_spar_num, itm_dens_num, profile_num, max_norm)
        with self.graph.as_default():
            self.d_reward = -tf.log(1 - self.y_pred + 1e-8)
            pred = self.pred + (self.seq_mask - 1) * 1e9
            self.build_discrim_loss(pred, c_entropy_d)

    def predict(self, item_spar_fts, item_dens_fts, seq_len):
        with self.graph.as_default():
            return self.sess.run([self.y_pred, self.d_reward], feed_dict={
                self.itm_spar_ph: item_spar_fts.reshape([-1, self.max_time_len, self.itm_spar_num]),
                self.itm_dens_ph: item_dens_fts.reshape([-1, self.max_time_len, self.itm_dens_num]),
                self.seq_length_ph: seq_len,
                self.keep_prob: 1.0,
                self.is_train: False})

    def train(self, batch_data, lr, reg_lambda, keep_prob=0.8):
        with self.graph.as_default():
            loss, _ = self.sess.run([self.loss, self.train_step], feed_dict={
                self.itm_spar_ph: batch_data[0].reshape([-1, self.max_time_len, self.itm_spar_num]),
                self.itm_dens_ph: batch_data[1].reshape([-1, self.max_time_len, self.itm_dens_num]),
                self.label_ph: batch_data[2].reshape([-1, self.max_time_len]),
                self.seq_length_ph: batch_data[3],
                self.lr: lr,
                self.reg_lambda: reg_lambda,
                self.keep_prob: keep_prob,
                self.is_train: True,
            })
            return loss


    def build_discrim_loss(self, logits, c_entropy_d):
        y_ = self.label_ph
        y = self.y_pred
        self.d_loss = -tf.reduce_mean(
            y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)) + (1 - y_) * tf.log(tf.clip_by_value(1 - y, 1e-10, 1.0)))

        self.entropy_loss = tf.reduce_mean(self.logit_bernoulli_entropy(logits))
        self.loss = self.d_loss - c_entropy_d * self.entropy_loss

        self.opt()

    def logit_bernoulli_entropy(self, logits):
        ent = (1. - tf.nn.sigmoid(logits)) * logits - self.logsigmoid(logits)
        return ent

    def logsigmoid(self, a):
        return -tf.nn.softplus(-a)


