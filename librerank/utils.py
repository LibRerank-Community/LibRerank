import pickle

import numpy as np
import pickle as pkl
from collections import defaultdict
import random
# from sklearn.metrics.pairwise import euclidean_distances
import argparse
import datetime
import json


def normalize(v):
    v = np.array(v)
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def softmax(x):
    x = x - np.max(x)
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x


def get_batch(data, batch_size, batch_no):
    return data[batch_size * batch_no: batch_size * (batch_no + 1)]


def repeat_data(data, rep_num):
    for i in range(len(data)):
        data_i = np.array(data[i])
        shape = data_i.shape
        tile_shape = np.ones_like(shape)
        tile_shape[-1] = rep_num
        new_shape = list(shape)
        new_shape[0] = -1
        data[i] = np.reshape(np.tile(data_i, tile_shape), new_shape).tolist()
    return data


def load_parse_from_json(parse, setting_path):
    with open(setting_path, 'r') as f:
        setting = json.load(f)
    parse_dict = vars(parse)
    for k, v in setting.items():
        parse_dict[k] = v
    return parse


def get_aggregated_batch(data, batch_size, batch_no):
    return [data[d][batch_size * batch_no: batch_size * (batch_no + 1)] for d in range(len(data))]


def padding_list(seq, max_len):
    spar_ft, dens_ft = seq
    seq_length = min(len(spar_ft), max_len)
    if len(spar_ft) < max_len or len(dens_ft) < max_len:
        spar_ft += [np.zeros_like(np.array(spar_ft[0])).tolist()] * (max_len - len(spar_ft))
        dens_ft += [np.zeros_like(np.array(dens_ft[0])).tolist()] * (max_len - len(dens_ft))
    return spar_ft[:max_len], dens_ft[:max_len], seq_length


def save_file(data, save_file):
    with open(save_file, 'w') as f:
        for v in data:
            line = '\t'.join([str(i) for i in v]) + '\n'
            f.write(line)


def load_file(save_file):
    with open(save_file, 'r') as f:
        data = f.readlines()
    records = []
    for line in data:
        records.append([eval(v) for v in line.split('\t')])
    return records


def construct_ranker_data(data):
    target_user, target_item_dens, target_item_spar, profiles, label, list_len = [], [], [], [], [], []
    # uids = []
    for i, d in enumerate(data):
        uid, profile, spar_ft, dens_ft, lb = d
        for j in range(len(lb)):
            target_user.append(i)
        # uids.append(uid)
        profiles.append(profile)
        target_item_dens.extend(dens_ft)
        target_item_spar.extend(spar_ft)
        label.extend(lb)
        list_len.append(len(lb))

    return target_user, profiles, target_item_spar, target_item_dens, label, list_len


def construct_behavior_data(data, max_len):
    target_user, target_item_dens, target_item_spar, user_behavior_dens, user_behavior_spar, label, seq_len, list_len, tiled_seq_len = [], [], [], [], [], [], [], [], []
    for d in data:
        uid, spar_ft, dens_ft, hist_spar, hist_dens, lb = d
        target_item_dens.extend(dens_ft)
        target_item_spar.extend(spar_ft)

        length = min(len(hist_spar), max_len)
        if len(hist_spar) < max_len:
            hist_spar = hist_spar + [np.zeros_like(np.array(hist_spar[0])).tolist()] * (max_len - len(hist_spar))
            hist_dens = hist_dens + [np.zeros_like(np.array(hist_dens[0])).tolist()] * (max_len - len(hist_dens))

        for i in range(len(lb)):
            target_user.append(uid)
            user_behavior_dens.append(hist_dens[:max_len])
            user_behavior_spar.append(hist_spar[:max_len])
            tiled_seq_len.append(length)
        label.extend(lb)
        seq_len.append(length)
        list_len.append(len(lb))
    print(target_item_spar[-1], label[-30:], len(target_item_spar), len(list_len), sum(list_len))

    return target_user, target_item_spar, target_item_dens, user_behavior_spar, user_behavior_dens, label, seq_len, list_len, tiled_seq_len


def rank(data, preds, out_file):
    users, profiles, item_spars, item_denss, labels, list_lens = data
    print('origin', item_spars[-1], labels[-30:], len(item_spars), len(list_lens), sum(list_lens), len(preds))
    out_user, out_itm_spar, out_itm_dens, out_label, out_pos = [], [], [], [], []
    idx = 0
    for i, length in enumerate(list_lens):
        item_spar, item_dens = item_spars[idx: idx + length], item_denss[idx: idx + length]
        # user_spar, user_dens = user_spars[idx], user_denss[idx]
        label, pred = labels[idx: idx + length], preds[idx: idx + length]
        rerank_idx = sorted(list(range(len(pred))), key=lambda k: pred[k], reverse=True)
        out_user.append(users[i])
        out_itm_spar.append(np.array(item_spar)[rerank_idx].tolist())
        out_itm_dens.append(np.array(item_dens)[rerank_idx].tolist())
        # out_usr_spar.append(user_spar)
        # out_usr_dens.append(user_dens)
        out_label.append(np.array(label)[rerank_idx].tolist())
        out_pos.append(np.arange(length)[rerank_idx].tolist())
        idx += length
    # print(len(out_itm_spar), out_label[-1], out_itm_spar[-1])
    with open(out_file, 'wb') as f:
        pickle.dump([out_user, profiles, out_itm_spar, out_itm_dens, out_label, out_pos, list_lens], f)


def get_last_click_pos(my_list):
    if sum(my_list) == 0 or sum(my_list) == len(my_list):
        return len(my_list) - 1
    return max([index for index, el in enumerate(my_list) if el])


def construct_list(data_dir, max_time_len):
    user, profile, itm_spar, itm_dens, label, pos, list_len = pickle.load(open(data_dir, 'rb'))
    print(len(user), len(itm_spar))
    cut_itm_dens, cut_itm_spar, cut_label, cut_pos, cut_usr_spar, cut_usr_dens, de_label, cut_hist_pos = [], [], [], [], [], [], [], []
    for i, itm_spar_i, itm_dens_i, label_i, pos_i, list_len_i in zip(list(range(len(label))),
                                                    itm_spar, itm_dens, label, pos, list_len):

        if len(itm_spar_i) >= max_time_len:
            cut_itm_spar.append(itm_spar_i[: max_time_len])
            cut_itm_dens.append(itm_dens_i[: max_time_len])
            cut_label.append(label_i[: max_time_len])
            # de_label.append(de_lb[: max_time_len])
            cut_pos.append(pos_i[: max_time_len])
            list_len[i] = max_time_len
        else:
            cut_itm_spar.append(itm_spar_i + [np.zeros_like(np.array(itm_spar_i[0])).tolist()] * (max_time_len - len(itm_spar_i)))
            cut_itm_dens.append(itm_dens_i + [np.zeros_like(np.array(itm_dens_i[0])).tolist()] * (max_time_len - len(itm_dens_i)))
            cut_label.append(label_i + [0 for _ in range(max_time_len - list_len_i)])
            # de_label.append(de_lb + [0 for _ in range(max_time_len - list_len_i)])
            cut_pos.append(pos_i + [j for j in range(list_len_i, max_time_len)])

    return user, profile, cut_itm_spar, cut_itm_dens, cut_label, cut_pos, list_len


def construct_list_with_profile(data_dir, max_time_len, max_seq_len, props, profile, use_pos=True):
    user, itm_spar, itm_dens, usr_spar, usr_dens, label, pos, list_len, seq_len = pickle.load(open(data_dir, 'rb'))
    print(len(user))
    print('max time len', max_time_len, 'max seq len', max_seq_len)
    max_interval, min_interval = 0, 1e9
    cut_itm_dens, cut_itm_spar, cut_label, cut_pos, cut_usr_spar, cut_usr_dens, de_label, user_prof, cut_hist_pos = [], [], [], [], [], [], [], [], []
    for i, itm_spar_i, itm_dens_i, usr_spar_i, usr_dens_i, label_i, pos_i, list_len_i, seq_len_i in zip(list(range(len(label))),
                                                    itm_spar, itm_dens, usr_spar, usr_dens, label, pos, list_len, seq_len):

        user_prof.append(profile[user[i]])
        de_lb = []
        for j in range(len(label_i)):
            de_lb.append(label_i[j] / props[itm_spar_i[j][1]][pos_i[j]])

        if len(itm_spar_i) >= max_time_len:
            cut_itm_spar.append(itm_spar_i[: max_time_len])
            cut_itm_dens.append(itm_dens_i[: max_time_len])
            cut_label.append(label_i[: max_time_len])
            de_label.append(de_lb[: max_time_len])
            cut_pos.append(pos_i[: max_time_len])
            list_len[i] = max_time_len
        else:

            cut_itm_spar.append(itm_spar_i + [np.zeros_like(np.array(itm_spar_i[0])).tolist()] * (max_time_len - len(itm_spar_i)))
            cut_itm_dens.append(itm_dens_i + [np.zeros_like(np.array(itm_dens_i[0])).tolist()] * (max_time_len - len(itm_dens_i)))
            # cut_itm_spar.append([np.zeros_like(np.array(itm_spar_i[0])).tolist()] * max_time_len)
            # cut_itm_dens.append([np.zeros_like(np.array(itm_dens_i[0])).tolist()] * max_time_len)
            cut_label.append(label_i + [0 for _ in range(max_time_len - list_len_i)])
            de_label.append(de_lb + [0 for _ in range(max_time_len - list_len_i)])
            cut_pos.append(pos_i + [j for j in range(list_len_i, max_time_len)])

        if len(usr_spar_i) >= max_seq_len:
            cut_usr_spar.append(usr_spar_i[:max_seq_len])
            cut_usr_dens.append(usr_dens_i[:max_seq_len])
            seq_len[i] = max_seq_len
        else:
            cut_usr_spar.append(
                usr_spar_i + [np.zeros_like(np.array(usr_spar_i[0])).tolist()] * (max_seq_len - len(usr_spar_i)))
            cut_usr_dens.append(
                usr_dens_i + [np.zeros_like(np.array(usr_dens_i[0])).tolist()] * (max_seq_len - len(usr_dens_i)))

        if use_pos:
            cut_hist_pos.append([j for j in range(max_seq_len)])
        else:
            usr_dens_i = np.log2(np.array(usr_dens_i) + 1)
            lst = np.reshape(np.array(usr_dens_i[:seq_len_i]), [-1]).tolist() #1

            hist_pos = lst + [max(lst) + 1 for i in range(seq_len_i, max_seq_len)]
            cut_hist_pos.append(hist_pos[:max_seq_len])

            max_interval = max(max_interval, max(cut_hist_pos[-1]))
            min_interval = min(min_interval, min(cut_hist_pos[-1]))

    print(max_interval, min_interval)

    return user_prof, cut_itm_spar, cut_itm_dens, cut_usr_spar, cut_usr_dens, cut_label, cut_hist_pos, list_len, seq_len, cut_pos, de_label


def get_sim_hist(profile_group, usr_profile):
    comm = profile_group[0][usr_profile[0]]
    idx = 1
    while idx < len(usr_profile):
        tmp = comm & profile_group[idx][usr_profile[idx]]
        idx += 1
        if not tmp:
            break
        comm = tmp
    return random.choice(list(comm))


def construct_list_with_profile_sim_hist(data_dir, max_time_len, max_seq_len, props, profile, profile_fnum, use_pos=True):
    user, itm_spar, itm_dens, usr_spar, usr_dens, label, pos, list_len, seq_len = pickle.load(open(data_dir, 'rb'))
    profile_group = [defaultdict(set) for _ in range(profile_fnum)]
    for u in range(len(user)):
        usr_prof = profile[user[u]]
        for i in range(profile_fnum):
            profile_group[i][usr_prof[i]].add(u)
    print(len(user))
    print('max time len', max_time_len, 'max seq len', max_seq_len)

    max_interval, min_interval = 0, 1e9
    cut_itm_dens, cut_itm_spar, cut_label, cut_pos, cut_usr_spar, cut_usr_dens, de_label, user_prof, cut_hist_pos = [], [], [], [], [], [], [], [], []
    for i, itm_spar_i, itm_dens_i, label_i, pos_i, list_len_i, seq_len_i in zip(list(range(len(label))),
                                                    itm_spar, itm_dens, label, pos, list_len, seq_len):

        user_prof.append(profile[user[i]])
        sim_id = get_sim_hist(profile_group, user_prof[-1])
        usr_dens_i, usr_spar_i = usr_dens[sim_id], usr_spar[sim_id]
        de_lb = []
        for j in range(len(label_i)):
            de_lb.append(label_i[j] / props[itm_spar_i[j][1]][pos_i[j]])

        if len(itm_spar_i) >= max_time_len:
            cut_itm_spar.append(itm_spar_i[: max_time_len])
            cut_itm_dens.append(itm_dens_i[: max_time_len])
            cut_label.append(label_i[: max_time_len])
            de_label.append(de_lb[: max_time_len])
            cut_pos.append(pos_i[: max_time_len])
            list_len[i] = max_time_len
        else:

            cut_itm_spar.append(itm_spar_i + [np.zeros_like(np.array(itm_spar_i[0])).tolist()] * (max_time_len - len(itm_spar_i)))
            cut_itm_dens.append(itm_dens_i + [np.zeros_like(np.array(itm_dens_i[0])).tolist()] * (max_time_len - len(itm_dens_i)))
            # cut_itm_spar.append([np.zeros_like(np.array(itm_spar_i[0])).tolist()] * max_time_len)
            # cut_itm_dens.append([np.zeros_like(np.array(itm_dens_i[0])).tolist()] * max_time_len)
            cut_label.append(label_i + [0 for _ in range(max_time_len - list_len_i)])
            de_label.append(de_lb + [0 for _ in range(max_time_len - list_len_i)])
            cut_pos.append(pos_i + [j for j in range(list_len_i, max_time_len)])


        if len(usr_spar_i) >= max_seq_len:
            cut_usr_spar.append(usr_spar_i[:max_seq_len])
            cut_usr_dens.append(usr_dens_i[:max_seq_len])
            seq_len[i] = max_seq_len
        else:
            cut_usr_spar.append(
                usr_spar_i + [np.zeros_like(np.array(usr_spar_i[0])).tolist()] * (max_seq_len - len(usr_spar_i)))
            cut_usr_dens.append(
                usr_dens_i + [np.zeros_like(np.array(usr_dens_i[0])).tolist()] * (max_seq_len - len(usr_dens_i)))

        if use_pos:
            cut_hist_pos.append([j for j in range(max_seq_len)])
        else:
            usr_dens_i = np.log2(np.array(usr_dens_i) + 1)
            lst = np.reshape(np.array(usr_dens_i[:seq_len_i]), [-1]).tolist() #1
            hist_pos = lst + [max(lst) + 1 for i in range(seq_len_i, max_seq_len)]
            cut_hist_pos.append(hist_pos[:max_seq_len])

            max_interval = max(max_interval, max(cut_hist_pos[-1]))
            min_interval = min(min_interval, min(cut_hist_pos[-1]))

    print(max_interval, min_interval)

    return user_prof, cut_itm_spar, cut_itm_dens, cut_usr_spar, cut_usr_dens, cut_label, cut_hist_pos, list_len, seq_len, cut_pos, de_label


def rerank(attracts, terms):
    val = np.array(attracts) * np.array(np.ones_like(terms))
    return sorted(range(len(val)), key=lambda k: val[k], reverse=True)


def evaluate(labels, preds, scope_number, props, cates, poss, is_rank):
    ndcg, utility, map, clicks = [], [], [], []
    for label, pred, cate, pos in zip(labels, preds, cates, poss):
        if is_rank:
            final = sorted(range(len(pred)), key=lambda k: pred[k], reverse=True)
        else:
            final = list(range(len(pred)))

        click = np.array(label)[final].tolist()  # reranked labels
        gold = sorted(range(len(click)), key=lambda k: click[k], reverse=True)  # optimal list for ndcg

        ideal_dcg, dcg, AP_value, AP_count, util = 0, 0, 0, 0, 0
        scope_number = min(scope_number, len(label))
        scope_gold = gold[:scope_number]

        for _i, _g, _f in zip(range(1, scope_number + 1), scope_gold, final[scope_number:]):
            dcg += (pow(2, click[_i - 1]) - 1) / (np.log2(_i + 1))
            ideal_dcg += (pow(2, click[_g]) - 1) / (np.log2(_i + 1))

            if click[_i] >= 1:
                AP_count += 1
                AP_value += AP_count / _i

            util += click[_i] * props[cate[_f]][_i]/props[cate[_f]][pos[_f]]

        _ndcg = float(dcg) / ideal_dcg if ideal_dcg != 0 else 0.
        _map = float(AP_value) / AP_count if AP_count != 0 else 0.

        ndcg.append(_ndcg)
        map.append(_map)
        utility.append(util)
        clicks.append(sum(clicks[:scope_number]))
    return np.mean(np.array(map)), np.mean(np.array(ndcg)), np.mean(np.array(clicks)), np.mean(np.array(utility)), \
            [map, ndcg, clicks, utility]



def evaluate_multi(labels, preds, scope_number, is_rank, _print=False):
    ndcg, map, clicks = [[] for _ in range(len(scope_number))], [[] for _ in range(len(scope_number))], [[] for _ in range(len(scope_number))]

    for label, pred in zip(labels, preds):
        if is_rank:
            final = sorted(range(len(pred)), key=lambda k: pred[k], reverse=True)
        else:
            final = list(range(len(pred)))
        click = np.array(label)[final].tolist()  # reranked labels
        gold = sorted(range(len(label)), key=lambda k: label[k], reverse=True)  # optimal list for ndcg

        for i, scope in enumerate(scope_number):
            ideal_dcg, dcg, de_dcg, de_idcg, AP_value, AP_count, util = 0, 0, 0, 0, 0, 0, 0
            cur_scope = min(scope, len(label))
            for _i, _g, _f in zip(range(1, cur_scope + 1), gold[:cur_scope], final[:cur_scope]):
                dcg += (pow(2, click[_i - 1]) - 1) / (np.log2(_i + 1))
                ideal_dcg += (pow(2, label[_g]) - 1) / (np.log2(_i + 1))

                if click[_i - 1] >= 1:
                    AP_count += 1
                    AP_value += AP_count / _i

            _ndcg = float(dcg) / ideal_dcg if ideal_dcg != 0 else 0.
            _map = float(AP_value) / AP_count if AP_count != 0 else 0.

            ndcg[i].append(_ndcg)
            map[i].append(_map)
            clicks[i].append(sum(click[:cur_scope]))

    return np.mean(np.array(map), axis=-1), np.mean(np.array(ndcg), axis=-1), np.mean(np.array(clicks), axis=-1), \
           [map, ndcg, clicks]
