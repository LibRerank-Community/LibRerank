import tensorflow as tf
import pickle as pkl
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from multiprocessing import Pool
import lightgbm as lgb



class BaseModel(object):
    def __init__(self, eb_dim, feature_size, itm_spar_fnum, itm_dens_fnum, user_fnum, num_item):
        # reset graph
        tf.reset_default_graph()

        # input placeholders
        with tf.name_scope('inputs'):
            self.user_behavior_length_ph = tf.placeholder(tf.int32, [None, ], name='user_behavior_length_ph')
            self.target_user_ph = tf.placeholder(tf.int32, [None, user_fnum], name='target_user_ph')
            self.item_spar_ph = tf.placeholder(tf.int32, [None, itm_spar_fnum], name='item_spar_ph')
            self.item_dens_ph = tf.placeholder(tf.float32, [None, itm_dens_fnum], name='item_dens_ph')
            self.label_ph = tf.placeholder(tf.int32, [None, ], name='label_ph')

            # lr
            self.lr = tf.placeholder(tf.float32, [])
            # reg lambda
            self.reg_lambda = tf.placeholder(tf.float32, [])
            # keep prob
            self.keep_prob = tf.placeholder(tf.float32, [])
            self.emb_dim = eb_dim
            self.itm_spar_fnum = itm_spar_fnum
            self.itm_dens_fnum = itm_dens_fnum
            self.user_fnum = user_fnum
            # self.num_user = num_user
            self.num_item = num_item
            # self.ft_len = itm_spar_fnum * eb_dim + itm_dens_fnum
            self.ft_len = itm_spar_fnum * eb_dim


        # embedding
        with tf.name_scope('embedding'):
            self.emb_mtx = tf.get_variable('emb_mtx', [feature_size, eb_dim],
                                           initializer=tf.truncated_normal_initializer)
            self.target_item = tf.nn.embedding_lookup(self.emb_mtx, self.item_spar_ph)
            # self.user_seq = tf.nn.embedding_lookup(self.emb_mtx, self.user_spar_ph)

            self.target_item = tf.reshape(self.target_item, [-1, itm_spar_fnum * eb_dim])
            self.target_item = tf.concat([self.target_item, self.item_dens_ph], axis=-1)

    def build_fc_net(self, inp):
        bn1 = tf.layers.batch_normalization(inputs=inp, name='bn1')
        fc1 = tf.layers.dense(bn1, 200, activation=tf.nn.relu, name='fc1')
        dp1 = tf.nn.dropout(fc1, self.keep_prob, name='dp1')
        fc2 = tf.layers.dense(dp1, 80, activation=tf.nn.relu, name='fc2')
        dp2 = tf.nn.dropout(fc2, self.keep_prob, name='dp2')
        fc3 = tf.layers.dense(dp2, 2, activation=None, name='fc3')
        score = tf.nn.softmax(fc3)
        # output
        self.y_pred = tf.reshape(score[:, 0], [-1, ])

    def build_mlp_net(self, inp, layer=(500, 200, 80)):
        for i, hidden_num in enumerate(layer):
            bn = tf.layers.batch_normalization(inputs=inp, name='bn' + str(i))
            fc = tf.layers.dense(bn, hidden_num, activation=tf.nn.relu, name='fc' + str(i))
            inp = tf.nn.dropout(fc, self.keep_prob, name='dp' + str(i))
        final = tf.layers.dense(inp, 2, activation=None, name='fc_final')
        score = tf.nn.softmax(final)
        # output
        self.y_pred = tf.reshape(score[:, 0], [-1, ])

    def build_logloss(self):
        # loss
        self.log_loss = tf.losses.log_loss(self.label_ph, self.y_pred)
        self.loss = self.log_loss
        for v in tf.trainable_variables():
            if 'bias' not in v.name and 'emb' not in v.name:
                self.loss += self.reg_lambda * tf.nn.l2_loss(v)
        # optimizer and training step
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.train_step = self.optimizer.minimize(self.loss)

    def build_mseloss(self):
        self.loss = tf.losses.mean_squared_error(self.label_ph, self.y_pred)
        # regularization term
        for v in tf.trainable_variables():
            if 'bias' not in v.name and 'emb' not in v.name:
                self.loss += self.reg_lambda * tf.nn.l2_loss(v)
        # optimizer and training step
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.train_step = self.optimizer.minimize(self.loss)

    def train(self, sess, batch_data, lr, reg_lambda):
        loss, _ = sess.run([self.loss, self.train_step], feed_dict={
            self.target_user_ph: np.array(batch_data[1]).reshape(-1, self.user_fnum),
            self.item_spar_ph: np.array(batch_data[2]),
            self.item_dens_ph: np.array(batch_data[3]),
            self.label_ph: np.array(batch_data[4]),
            # self.user_behavior_length_ph: np.array(batch_data[-1]),
            self.lr: lr,
            self.reg_lambda: reg_lambda,
            self.keep_prob: 0.8,
        })
        return loss

    def eval(self, sess, batch_data, reg_lambda):

        pred, label, loss = sess.run([self.y_pred, self.label_ph, self.loss], feed_dict={
            self.target_user_ph: np.array(batch_data[1]).reshape(-1, self.user_fnum),
            self.item_spar_ph: batch_data[2],
            self.item_dens_ph: batch_data[3],
            self.label_ph: batch_data[4],
            self.reg_lambda: reg_lambda,
            self.keep_prob: 1.
        })
        return pred.tolist(), label.tolist(), loss

    def save(self, sess, path):
        saver = tf.train.Saver()
        saver.save(sess, save_path=path)

    def save_pretrain(self, sess, path):
        embed = sess.run(self.emb_mtx)
        with open(path, 'wb') as f:
            pkl.dump(embed, f)


class DNN(BaseModel):
    def __init__(self, eb_dim, feature_size, itm_spar_fnum, itm_dens_fnum, user_fnum, num_item):
        super(DNN, self).__init__(eb_dim, feature_size, itm_spar_fnum, itm_dens_fnum, user_fnum, num_item)

        # inp = tf.concat([tf.reshape(self.user_seq, [-1, (itm_spar_fnum * eb_dim + itm_dens_fnum) * max_time_len]), self.target_item], axis=1)
        inp = self.target_item
        # fc layer
        # self.build_fc_net(inp)
        self.build_mlp_net(inp)
        self.build_logloss()


def dcg(scores):
    return np.sum([ (np.power(2, scores[i]) - 1) / np.log2(i + 2)for i in range(len(scores))])


def dcg_k(scores, k):
    return np.sum([ (np.power(2, scores[i]) - 1) / np.log2(i + 2)for i in range(len(scores[:k]))])


def ideal_dcg(scores):
    scores = [score for score in sorted(scores)[::-1]]
    return dcg(scores)


def ideal_dcg_k(scores, k):
    scores = [score for score in sorted(scores)[::-1]]
    return dcg_k(scores, k)


def single_dcg(scores, i, j):
    return (np.power(2, scores[i]) - 1) / np.log2(j + 2)


def compute_lambda(args):
    """
        Returns the lambda and w values for a given query.
        Parameters
        ----------
        args : zipped value of true_scores, predicted_scores, good_ij_pairs, idcg, query_key
            Contains a list of the true labels of documents, list of the predicted labels of documents,
            i and j pairs where true_score[i] > true_score[j], idcg values, and query keys.

        Returns
        -------
        lambdas : numpy array
            This contains the calculated lambda values
        w : numpy array
            This contains the computed w values
        query_key : int
            This is the query id these values refer to
    """

    true_scores, predicted_scores, good_ij_pairs, idcg, query_key = args

    num_docs = len(true_scores)
    sorted_indexes = np.argsort(-predicted_scores)
    rev_indexes = np.argsort(sorted_indexes)
    true_scores = true_scores[sorted_indexes]
    predicted_scores = predicted_scores[sorted_indexes]

    lambdas = np.zeros(num_docs)
    w = np.zeros(num_docs)

    # print("compute dcg----------------------------------")
    single_dcgs = {}
    for i, j in good_ij_pairs:
        if (i, i) not in single_dcgs:
            single_dcgs[(i, i)] = single_dcg(true_scores, i, i)
        single_dcgs[(i, j)] = single_dcg(true_scores, i, j)
        if (j, j) not in single_dcgs:
            single_dcgs[(j, j)] = single_dcg(true_scores, j, j)
        single_dcgs[(j, i)] = single_dcg(true_scores, j, i)

    # print("______________________________________________")

    for i, j in good_ij_pairs:
        z_ndcg = abs(
            single_dcgs[(i, j)]- single_dcgs[(i, i)]+ single_dcgs[(j, i)]- single_dcgs[(j, j)]) / idcg
        rho = 1 / (1 + np.exp(predicted_scores[i] - predicted_scores[j]))
        rho_complement = 1.0 - rho
        lambda_val = z_ndcg * rho
        lambdas[i] += lambda_val
        lambdas[j] -= lambda_val

        w_val = rho * rho_complement * z_ndcg
        w[i] += w_val
        w[j] += w_val

    return lambdas[rev_indexes], w[rev_indexes], query_key


def group_queries(training_data, qid_index):
    query_indexes = {}
    index = 0
    for record in training_data:
        query_indexes.setdefault(record[qid_index], [])
        query_indexes[record[qid_index]].append(index)
        index += 1
    return query_indexes


def get_pairs(true_scores, pred_scores):
    query_pair = []
    for i, query_scores in enumerate(pred_scores):
        sorted_indexes = np.argsort(-np.array(query_scores))
        temp = np.array(true_scores[i])[sorted_indexes].tolist()
        pairs = []
        for i in range(len(temp)):
            for j in range(len(temp)):
                if temp[i] > temp[j]:
                    pairs.append((i, j))
        query_pair.append(pairs)
    return query_pair


class LambdaMART:

    def __init__(self, training_data=None, number_of_trees=5, learning_rate=0.1, tree_type='sklearn'):
        """
        This is the constructor for the LambdaMART object.
        Parameters
        ----------
        training_data : list of int
            Contain a list of numbers
        number_of_trees : int (default: 5)
            Number of trees LambdaMART goes through
        learning_rate : float (default: 0.1)
            Rate at which we update our prediction with each tree
        tree_type : string (default: "sklearn")
            Either "sklearn" for using Sklearn implementation of the tree of "original"
            for using our implementation
        """

        if tree_type != 'sklearn' and tree_type != 'lgb':
            raise ValueError('The "tree_type" must be "sklearn" or "lgb"')
        self.training_data = np.array(training_data)
        # self.test_data = test_data
        self.number_of_trees = number_of_trees
        self.learning_rate = learning_rate
        self.trees = []
        self.tree_type = tree_type

    def fit(self):
        """
        Fits the model on the training data.
        """
        # self.test_data = test_dataset
        predicted_scores = np.zeros(self.training_data.shape[0])

        query_indexes = group_queries(self.training_data, 1)
        # test_query_indexes = group_queries(self.test_data, 1)
        query_keys = list(query_indexes.keys())
        # test_query_key = list(test_query_indexes.keys())
        true_scores = [self.training_data[query_indexes[query], 0] for query in query_keys]
        # exam = [train_exam[query_indexes[query]] for query in query_keys]

        # tree_data = pd.DataFrame(self.training_data[:, 3:])
        # labels = self.training_data[:, 0]

        # ideal dcg calculation
        idcg = [ideal_dcg(scores) for scores in true_scores]

        for k in range(self.number_of_trees):
            print('Tree %d' % (k))
            lambdas = np.zeros(len(predicted_scores))
            w = np.zeros(len(predicted_scores))
            pred_scores = [predicted_scores[query_indexes[query]] for query in query_keys]
            good_ij_pairs = get_pairs(true_scores, pred_scores)

            pool = Pool()
            for lambda_val, w_val, query_key in pool.map(compute_lambda,
                                                         zip(true_scores, pred_scores, good_ij_pairs, idcg,
                                                             query_keys), chunksize=1):
                indexes = query_indexes[query_key]
                lambdas[indexes] = lambda_val
                w[indexes] = w_val
            pool.close()
            pool.join()

            if self.tree_type == 'sklearn':
                # Sklearn implementation of the tree
                tree = DecisionTreeRegressor(splitter="random")
                tree.fit(self.training_data[:, 2:], lambdas)
                self.trees.append(tree)
                prediction = tree.predict(self.training_data[:, 2:])
                predicted_scores += prediction * self.learning_rate

            elif self.tree_type == 'lgb':
                gbm = lgb.LGBMRegressor(objective='regression', num_leaves=31, learning_rate=0.1, n_estimators=10)
                gbm.fit(self.training_data[:, 2:], lambdas)
                self.trees.append(gbm)
                prediction = gbm.predict(self.training_data[:, 2:])
                predicted_scores += prediction * self.learning_rate

    def predict(self, test_dataset):
        """
        Predicts the scores for the test dataset.
        Parameters
        ----------
        data : Numpy array of documents
            Numpy array of documents with each document's format is [query index, feature vector]

        Returns
        -------
        predicted_scores : Numpy array of scores
            This contains an array or the predicted scores for the documents.
        """
        data = np.array(test_dataset)
        query_indexes = group_queries(data, 1)
        predicted_scores = np.zeros(len(data))
        for query in query_indexes:
            results = np.zeros(len(query_indexes[query]))
            for tree in self.trees:
                results += self.learning_rate * tree.predict(data[query_indexes[query], 2:])
            predicted_scores[query_indexes[query]] = results
        return predicted_scores

    def save(self, fname):
        """
        Saves the model into a ".lmart" file with the name given as a parameter.
        Parameters
        ----------
        fname : string
            Filename of the file you want to save

        """
        pkl.dump(self, open('%s.mart' % (fname), "wb"), protocol=2)

    def load(self, fname):
        """
        Loads the model from the ".lmart" file given as a parameter.
        Parameters
        ----------
        fname : string
            Filename of the file you want to load

        """
        model = pkl.load(open(fname, "rb"))
        self.training_data = model.training_data
        self.number_of_trees = model.number_of_trees
        self.tree_type = model.tree_type
        self.learning_rate = model.learning_rate
        self.trees = model.trees