import os
import tensorflow as tf
from sklearn.metrics import log_loss, roc_auc_score
import time

from librerank.utils import *
from librerank.ranker import LambdaMART, DNN


def eval(model, sess, data, reg_lambda, batch_size):
    preds = []
    labels = []
    losses = []

    data_size = len(data[0])
    batch_num = data_size // batch_size
    print('eval', data_size, batch_size, batch_num)
    t = time.time()
    for batch_no in range(batch_num):
        data_batch = get_aggregated_batch(data, batch_size=batch_size, batch_no=batch_no)

        pred, label, loss = model.eval(sess, data_batch, reg_lambda)
        preds.extend(pred)
        labels.extend(label)
        losses.append(loss)

    logloss = log_loss(labels, preds)
    auc = roc_auc_score(labels, preds)
    loss = sum(losses) / len(losses)

    print("EVAL TIME: %.4fs" % (time.time() - t))
    return loss, logloss, auc


def save_rank(model, sess, data, reg_lambda, batch_size, out_file):
    preds = []

    data_size = len(data[0])
    batch_num = data_size // batch_size
    if data_size % batch_size:
        batch_num += 1
    print(data_size, batch_size, batch_num)

    for batch_no in range(batch_num):
        data_batch = get_aggregated_batch(data, batch_size=batch_size, batch_no=batch_no)
        pred, label, loss = model.eval(sess, data_batch, reg_lambda)
        preds.extend(pred)
    # print('pred', len(preds))
    rank(data, preds, out_file)


def train(train_file, val_file, test_file, eb_dim, feature_size, itm_spar_fnum,
          itm_dens_fnum, user_fnum, num_item, lr, reg_lambda, batch_size, processed_dir, pt_dir):
    tf.reset_default_graph()

    if parse.model_type == 'DNN':
        model = DNN(eb_dim, feature_size, itm_spar_fnum, itm_dens_fnum, user_fnum, num_item)
    else:
        print('WRONG MODEL TYPE')
        exit(1)

    training_monitor = {
        'train_loss': [],
        'vali_loss': [],
        'logloss': [],
        'auc': []
    }

    # gpu settings
    gpu_options = tf.GPUOptions(allow_growth=True)

    # training process
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        train_losses_step = []

        # before training process
        step = 0
        vali_loss, logloss, auc = eval(model, sess, test_file, reg_lambda, batch_size)

        training_monitor['train_loss'].append(None)
        training_monitor['vali_loss'].append(vali_loss)
        training_monitor['logloss'].append(logloss)
        training_monitor['auc'].append(auc)

        print("STEP %d  LOSS TRAIN: NULL | LOSS VALI: %.4f  LOGLOSS: %.4f  AUC: %.4f" % (
        step, vali_loss, logloss, auc))
        early_stop = False
        data_size = len(train_file[0])
        batch_num = data_size // batch_size
        eval_iter_num = (data_size // 10) // batch_size
        print('train', data_size, batch_num)

        # begin training process
        for epoch in range(parse.epoch_num):
            if early_stop:
                break
            for batch_no in range(batch_num):
                data_batch = get_aggregated_batch(train_file, batch_size=batch_size, batch_no=batch_no)
                # if early_stop:
                #     break
                loss = model.train(sess, data_batch, lr, reg_lambda)


                step += 1
                train_losses_step.append(loss)

                if step % eval_iter_num == 0:
                    train_loss = sum(train_losses_step) / len(train_losses_step)
                    training_monitor['train_loss'].append(train_loss)
                    train_losses_step = []

                    vali_loss, logloss, auc = eval(model, sess, test_file, reg_lambda, batch_size)
                    training_monitor['vali_loss'].append(vali_loss)
                    training_monitor['logloss'].append(logloss)
                    training_monitor['auc'].append(auc)

                    print("EPOCH %d  STEP %d  LOSS TRAIN: %.4f | LOSS VALI: %.4f  LOGLOSS: %.4f  AUC: %.4f" % (
                    epoch, step, train_loss, vali_loss, logloss, auc))
                    if training_monitor['auc'][-1] > max(training_monitor['auc'][:-1]):
                        # save model
                        model_name = '{}_{}_{}_{}'.format(parse.model_type, batch_size, lr, reg_lambda)
                        if not os.path.exists('{}/save_model_{}/ranker/{}/'.format(parse.save_dir, data_set_name, model_name)):
                            os.makedirs('{}/save_model_{}/ranker/{}/'.format(parse.save_dir, data_set_name, model_name))
                        save_path = '{}/save_model_{}/ranker/{}/ckpt'.format(parse.save_dir, data_set_name, model_name)
                        model.save(sess, save_path)
                        save_rank(model, sess, val_file, reg_lambda, batch_size, processed_dir + parse.model_type + '.rankings.train')
                        save_rank(model, sess, test_file, reg_lambda, batch_size, processed_dir + parse.model_type + '.rankings.test')
                        model.save_pretrain(sess, pt_dir)
                        print('intial lists saved')
                        early_stop = False
                        continue

                    if len(training_monitor['auc']) > 2 and epoch > 0:
                        # if (training_monitor['vali_loss'][-1] > training_monitor['vali_loss'][-2] and
                        #         training_monitor['vali_loss'][-2] > training_monitor['vali_loss'][-3]):
                        #     early_stop = True
                        if (training_monitor['auc'][-2] - training_monitor['auc'][-1]) >= 0.1 and (
                                training_monitor['auc'][-3] - training_monitor['auc'][-2]) >= 0.1:
                            early_stop = True

        # generate log
        if not os.path.exists('{}/logs_{}/ranker/'.format(parse.save_dir, data_set_name)):
            os.makedirs('{}/logs_{}/ranker/'.format(parse.save_dir, data_set_name))
        model_name = '{}_{}_{}_{}_{}'.format(parse.timestamp, parse.model_type, batch_size, lr, reg_lambda)

        with open('{}/logs_{}/ranker/{}.pkl'.format(parse.save_dir, data_set_name, model_name), 'wb') as f:
            pkl.dump(training_monitor, f)


def get_data(dataset, embed_dir):
    users, profiles, item_spars, item_denss, labels, list_lens = dataset
    embeddings = pkl.load(open(embed_dir, 'rb'))
    embeddings = embeddings
    records = []
    uid_idx = 0
    # uid_map = {}
    for itm_spar_i, itm_dens_i, label_i, uid_i in zip(item_spars, item_denss, labels, users):
        itm_emd = np.reshape(np.array(embeddings[itm_spar_i]), -1)
        # record_i = [label_i, uid_i] + itm_emd.tolist() + itm_dens_i
        record_i = [label_i, uid_i] + itm_emd.tolist()
        uid_idx += 1
        records.append(record_i)
    return records, np.reshape(np.array(labels), -1).tolist()


def train_mart(train_file, val_file, test_file, embed_dir, processed_dir,
               tree_num=300, lr=0.05, tree_type='lgb'):
    training_data, labels = get_data(train_file, embed_dir)
    model = LambdaMART(training_data, tree_num, lr, tree_type)
    model.fit()
    if not os.path.exists('{}/save_model_{}/ranker/'.format(parse.save_dir, data_set_name)):
        os.makedirs('{}/save_model_{}/ranker/'.format(parse.save_dir, data_set_name))
    # model.save('{}/save_model_{}/ranker/{}_{}_{}_{}'.format(parse.save_dir, data_set_name, parse.timestamp, tree_num, lr, tree_type))
    training_data = []

    print('test set')
    test_data, labels = get_data(test_file, embed_dir)
    test_pred = model.predict(test_data)
    rank(test_file, test_pred, processed_dir + parse.model_type +'.rankings.test')
    logloss = log_loss(labels, test_pred)
    auc = roc_auc_score(labels, test_pred)
    print('mart logloss:', logloss, 'auc:', auc)
    test_data = []

    print('valid set')
    val_data, labels = get_data(val_file, embed_dir)
    val_pred = model.predict(val_data)
    rank(val_file, val_pred, processed_dir + parse.model_type + '.rankings.train')


def save_svm_file(dataset, out_file):
    svm_rank_fout = open(out_file, 'w')
    for i, record in enumerate(dataset):
        feats = []
        for j, v in enumerate(record[2:]):
            feats.append(str(j + 1) + ':' + str(v))
        line = str(int(record[0])) + ' qid:' + str(int(record[1])) + ' ' + ' '.join(feats) + '\n'
        svm_rank_fout.write(line)
    svm_rank_fout.close()


def train_svm(train_file, val_file, test_file, embed_dir, processed_dir, c=2.0):
    svm_dir = processed_dir + 'svm'
    if not os.path.exists(svm_dir):
        os.makedirs(svm_dir)
    training_data, train_labels = get_data(train_file, embed_dir)
    save_svm_file(training_data, svm_dir + '/train.txt')
    training_data, train_labels = [], []
    test_data, test_labels = get_data(test_file, embed_dir)
    save_svm_file(test_data, svm_dir + '/test.txt')
    test_data = []
    val_data, val_labels = get_data(val_file, embed_dir)
    save_svm_file(val_data, svm_dir + '/valid.txt')
    val_data, val_labels = [], []

    # train SVMrank model
    command = 'SVMrank/svm_rank_learn -c ' + str(c) + ' ' + svm_dir + '/train.txt ' + svm_dir + '/model.dat'
    os.system(command)

    # test the train set left, generate initial rank for context feature and examination
    # SVM_rank_path+svm_rank_classify remaining_train_set_path output_model_path output_prediction_path
    command = 'SVMrank/svm_rank_classify ' + svm_dir + '/test.txt ' + svm_dir + '/model.dat ' + svm_dir + '/test.predict'
    os.system(command)
    command = 'SVMrank/svm_rank_classify ' + svm_dir + '/valid.txt ' + svm_dir + '/model.dat ' + svm_dir + '/valid.predict'
    os.system(command)

    test_fin = open(svm_dir + '/test.predict', 'r')
    test_pred = list(map(float, test_fin.readlines()))
    test_fin.close()
    rank(test_file, test_pred, processed_dir + parse.model_type + '.rankings.test')
    logloss = log_loss(test_labels, test_pred)
    auc = roc_auc_score(test_labels, test_pred)
    print('mart logloss:', logloss, 'auc:', auc)

    val_fin = open(svm_dir + '/valid.predict', 'r')
    val_pred = list(map(float, val_fin.readlines()))
    val_fin.close()
    rank(val_file, val_pred,  processed_dir + parse.model_type + '.rankings.train')


def ranker_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data/toy/', help='the path of data')
    parser.add_argument('--save_dir', type=str, default='./', help='dir that saves logs and model')
    parser.add_argument('--model_type', default='mart', choices=['DNN', 'mart'], type=str,
                        help='algorithm name, including DNN, mart, svm')
    parser.add_argument('--data_set_name', default='ad', type=str, help='name of dataset')
    parser.add_argument('--epoch_num', default=50, type=int, help='epochs of each iteration.')
    parser.add_argument('--batch_size', default=500, type=int, help='batch size')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--l2_reg', default=1e-5, type=float, help='l2 loss scale')
    parser.add_argument('--eb_dim', default=16, type=int, help='size of embedding')
    parser.add_argument('--hidden_size', default=64, type=int, help='hidden size')
    parser.add_argument('--tree_type', default='lgb', type=str, help='tree type for lambdamart')
    parser.add_argument('--tree_num', default=10, type=int, help='num of tree for lambdamart')
    parser.add_argument('--c', default=2, type=float, help='c for SVM')
    parser.add_argument('--decay_steps', default=3000, type=int, help='learning rate decay steps')
    parser.add_argument('--decay_rate', default=1.0, type=float, help='learning rate decay rate')
    parser.add_argument('--timestamp', type=str, default=datetime.datetime.now().strftime("%Y%m%d%H%M"))
    parser.add_argument('--reload_path', type=str, default='', help='model ckpt dir')
    parser.add_argument('--setting_path', type=str, default='', help='setting dir')
    FLAGS, _ = parser.parse_known_args()
    return FLAGS


if __name__ == '__main__':
    # parameters
    random.seed(1234)
    parse = ranker_parse_args()
    if parse.setting_path:
        parse = load_parse_from_json(parse, parse.setting_path)
    # data_dir = 'data/'
    data_set_name = parse.data_set_name
    # data_set_name = 'ad'
    processed_dir = parse.data_dir
    stat_dir = os.path.join(processed_dir, 'data.stat')
    pt_dir = os.path.join(processed_dir, 'pretrain')

    # model_type = 'DIN'
    # model_type = 'svm'
    # model_type = 'DNN'
    # model_type = 'mart'
    #
    # reg_lambda = 1e-5
    # lr = 1e-4 #DIN
    # # lr = 5e-5
    # embedding_size = 16
    # batch_size = 500
    # tree_num = 1
    # tree_type = 'lgb'
    # c = 0.2
    # epoch_num = 50

    with open(stat_dir, 'r') as f:
        stat = json.load(f)

    num_item, num_cate, num_ft, profile_fnum, itm_spar_fnum, itm_dens_fnum, = stat['item_num'], stat['cate_num'], \
        stat['ft_num'], stat['profile_fnum'], stat['itm_spar_fnum'], stat['itm_dens_fnum']

    print('num of item', num_item, 'num of list', stat['train_num'] + stat['val_num'] + stat['test_num'],
          'profile num', profile_fnum, 'spar num', itm_spar_fnum, 'dens num', itm_dens_fnum)

    data = load_file(os.path.join(processed_dir, 'data.train'))
    train_file = construct_ranker_data(data)
    data = load_file(os.path.join(processed_dir, 'data.valid'))
    val_file = construct_ranker_data(data)
    data = load_file(os.path.join(processed_dir, 'data.test'))
    test_file = construct_ranker_data(data)
    data = []


    if parse.model_type == 'DNN':
        train(train_file, val_file, test_file, parse.eb_dim, num_ft,
              itm_spar_fnum, itm_dens_fnum, profile_fnum, num_item, parse.lr,
              parse.l2_reg, parse.batch_size, processed_dir, pt_dir)
    elif parse.model_type == 'mart':
        train_mart(train_file, val_file, test_file, pt_dir,
              processed_dir, parse.tree_num, parse.lr, parse.tree_type)
    # elif parse.model_type == 'svm':
    #     train_svm(train_file, val_file, test_file, pt_dir, processed_dir, parse.c)
    else:
        print('No Such Model')




