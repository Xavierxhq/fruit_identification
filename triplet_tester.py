import torch
from models.networks import get_baseline_model
from datasets import data_loader
from torch.autograd import Variable
from utils.transforms import TestTransform
import numpy as np
import os
import re
import sys
import shutil
import pickle
import time
from get_xls_from_map import init_dict

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.cuda.manual_seed(0)

dish_map_dict = {  # we need the dict.
    '32':[0, 10],
    '35': [4],
    '12': [4],
    '26': [4],
    '50': [1],
    '21': [4],
    '29': [9],
    '8': [4],
    '4': [5],
    '3': [9],
    '47': [4],
    '38': [9],
    '42': [5],
    '33': [5],
    '28': [1],
    '16': [9],
    '43': [9],
    '15': [9],
    '6': [4],
    '37': [4],
    '17': [9],
    '11': [4],
    '18': [4],
    '41': [8],
    '5': [1],
    '14': [4],
    '24': [8],
    '46': [5],
    '22': [6],
    '48': [4],
    '27': [2],
    '44': [9],
    '54': [4],
    '9': [4],
    '1': [9],
    '30': [4],
    '23': [9],
    '40': [5],
    '53': [4],
    '2': [4],
    '31': [9],
    '10': [1],
    '51': [3],
    '13': [8],
    '36': [9],
    '34': [1],
    '20': [10, 4],
    '45': [11],
    '39': [9],
    '19': [5],
    '7': [5],
    '25': [4],
    '52': [1],
    '49': [4]
}


def dist(y1, y2):  # ok
    y2 = y2.cuda()
    return torch.sqrt(torch.sum(torch.pow(y1 - y2, 2))).item()


def get_proper_input(img_path):  # ok
    if not os.path.exists(img_path):
        return None
    pic_data = data_loader.read_image(img_path)
    lst = list()
    HEIGHT = 128
    WIDTH = 128
    test = TestTransform(WIDTH, HEIGHT)
    lst.append(np.array(test(pic_data)))
    lst = np.array(lst)
    pic_data = Variable(torch.from_numpy(lst))
    return pic_data


def get_feature(img_path, base_model, use_cuda=True):  # ok
    x = get_proper_input(img_path)
    if use_cuda:
        x = x.cuda()
    y = base_model(x)
    if use_cuda:
        y = y.cuda()
    return y


def get_dis(img_path_1, img_path_2, base_model):  # ok
    y1 = get_feature(img_path_1, base_model)
    y2 = get_feature(img_path_2, base_model)
    return dist(y1, y2)


def load_model(model_path=None, layers=50):
    if not model_path:
        model_path = 'model/resnet50-19c8e357.pth'
    base_model, optim_policy = get_baseline_model(model_path=model_path, layers=layers)
    model_parameter = torch.load(model_path)
    base_model.load_state_dict(model_parameter['state_dict'])
    base_model = base_model.cuda()
    print('model', model_path.split('/')[-1], 'loaded.')
    return base_model


def evaluate_single_file_with_average_feature_map(file_path, feature_map, base_model):
    # start_time = time.time()
    # file_path: single picture path
    result_dict = {}
    file_feature = get_feature(file_path, base_model)
    # print(file_feature)
    # print(feature_map)
    for k, v in feature_map.items():
        if type(v) == dict:
            continue
        _feature = torch.FloatTensor(v)
        result_dict[k] = dist(file_feature, _feature)

    for k, v in result_dict.items():
        # result_dict[k] = np.asarray(v.detach().numpy())
        for i in np.nditer(result_dict[k]):
            result_dict[k] = float(str(i))

    my_map = sorted(result_dict.items(), key=lambda d: d[1])

    new_map = dict()
    rank_list = list()
    for i in range(len(my_map)):
        new_map[str(my_map[i][0])] = i
        rank_list.append(str(my_map[i][0]))
    # print('time for evaluate_single_file_with_average_feature_map:', '%.1f' % (time.time() - start_time), 's')
    # exit(200)

    return new_map, rank_list


def transform_feature_map_to_everage(origin_feature_map, output_map_path = '', feature_num_each_class=5, _range=55):
    new_feature_map = init_dict(_range, _range, 0)
    lst = [i for i in range(_range)]
    for index in lst:
        index = str(index)
        if str(index) in origin_feature_map:
            _avg_feature = np.zeros(shape=origin_feature_map[index][0].shape)
            for _feature in origin_feature_map[index]:
                _feature = _feature.cpu().detach().numpy()
                _avg_feature += _feature
            _avg_feature /= feature_num_each_class
            new_feature_map[index] = torch.FloatTensor(_avg_feature)
    pickle_write(output_map_path, new_feature_map)
    return new_feature_map


def get_feature_map_k(base_model, lst, sample_num_each_cls=5, margin=5, epoch=1, test_file_dir='datas/test_chawdoe/sample_data_', save_dir='evaluate_result/feature_map'):
    print('do get_feature_map')
    feature_map = dict()
    # lst is a list which includes class index as int array.
    test_file_dir += str(sample_num_each_cls)
    if not os.path.exists(test_file_dir):
        print('You must use get sample_std_file() firstly')
        return None

    for i in lst:
        ground_truth_label = str(i)
        feature_map[ground_truth_label] = list()
        dir_full_path = os.path.join(test_file_dir, ground_truth_label)  # open the directory in order.
        dir_file_list = os.listdir(dir_full_path)
        for file_name in dir_file_list:
            file_full_path = os.path.join(dir_full_path, file_name)
            if len(feature_map[ground_truth_label]) < sample_num_each_cls:
                feature_map[ground_truth_label].append(get_feature(file_full_path, base_model))

    save_file_name = 'margin({})_epoch({})_featureMap_{}_{}.pkl'.format(margin, epoch, len(lst), sample_num_each_cls)
    save_path = os.path.join(save_dir, save_file_name)

    pickle_write(save_path, feature_map)
    print('feature map has been saved in ' + save_path)
    return feature_map

def get_feature_map_average(base_model, lst, sample_num_each_cls=5, margin=5, epoch=1, test_file_dir='./base_sample/', save_dir='evaluate_result/feature_map'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_file_name = 'margin({})_epoch({})_featureMap_{}_{}.pkl'.format(margin, epoch, len(lst), sample_num_each_cls)
    save_path = os.path.join(save_dir, save_file_name)
    if os.path.exists(save_path):
        os.remove(save_path)
    start_time = time.time()
    # lst is a list which includes class index as int array.
    test_file_dir += str(sample_num_each_cls)
    if not os.path.exists(test_file_dir):
        print('You must use get sample_std_file() firstly')
        return None

    for i in lst:
        # print(i)
        ground_truth_label = str(i)
        # feature_map[ground_truth_label] = list()
        features = []
        dir_full_path = os.path.join(test_file_dir, ground_truth_label)  # open the directory in order.
        dir_file_list = os.listdir(dir_full_path)
        for file_name in dir_file_list:
            file_full_path = os.path.join(dir_full_path, file_name)
            # if len(feature_map[ground_truth_label]) < sample_num_each_cls:
            if len(features) < sample_num_each_cls:
                feature_on_gpu = get_feature(file_full_path, base_model)
                # feature_map[ground_truth_label].append(f)
                features.append(feature_on_gpu)
        write_feature_map(save_path, ground_truth_label, features)
        features = None

    feature_map = pickle_read(save_path)
    new_feature = transform_feature_map_to_everage(feature_map, save_path, _range=55)
    # print('feature map of avg has been saved in ' + save_path)
    print('time for generating feature map:', '%.1f' % (time.time() - start_time), 's')
    return new_feature


def write_feature_map(feature_map_name, label, features):
    if os.path.exists(feature_map_name):
        obj = pickle_read(feature_map_name)
        obj[label] = features
    else:
        obj = {
            label: features
        }
    pickle_write(feature_map_name, obj)


def get_sample_std_file(sample_num_each_cls=5, directory='./test_data/', save_dir_path='./base_sample/'):
    # half complete. Usually only use once in your first training.
    save_dir_path += str(sample_num_each_cls)

    if os.path.exists(save_dir_path):
        return

    sample_list, copy_file_name_list, sample_num_dict = [], [], {}

    for i in os.listdir(directory):
        line_list = re.split('_', i)
        class_index = line_list[-1][:-4]
        if class_index not in sample_num_dict:
            sample_num_dict[class_index] = 1
        elif sample_num_dict[class_index] == sample_num_each_cls:
            continue
        else:
            sample_num_dict[class_index] += 1
        sample_list.append(os.path.join(directory, i))
        copy_file_name_list.append(i)

    for i in range(len(sample_list)):
        line_list = re.split('_', copy_file_name_list[i])
        class_index = line_list[-1][:-4]
        save_dir = os.path.join(save_dir_path, class_index)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, copy_file_name_list[i])
        shutil.copyfile(sample_list[i], save_path)


def evaluate_single_file_knn(target_file_path, feature_map, base_model, ground_truth_label, base_num, top_times=2):
    # print('do evaluate_single_file, this time ground truth lable is:', ground_truth_label)
    target_feature = get_feature(target_file_path, base_model)
    distance_dict, top_num = {}, int(base_num * top_times)
    # k == top_num
    # in the following loop, we compute the distance between target feature and each base feature
    for k, v in feature_map.items():
        for base_feature in v:
            dis = dist(target_feature, base_feature)
            key = '%s_%f' % (k, time.time())
            distance_dict[key] = dis
    distance_dict = sorted(distance_dict.items(), key=lambda item: item[1])
    # do counting for the top rank distances, and to sort out every class's number
    distance_rank_dict = dict()
    for _i in range(top_num):
        key, _v = distance_dict[_i]
        key = key.split('_')[0]
        # print('key:', key, ', value:', _v)
        if key not in distance_rank_dict:    # k == top_num
            distance_rank_dict[key] = 1
        else:
            distance_rank_dict[key] += 1
        if distance_rank_dict[key] > int(top_num / 2) or (distance_rank_dict[key] == ground_truth_label and distance_rank_dict[key] == base_num):
            # print('key:', key, 'count for now:', distance_rank_dict[key])
            return key, distance_rank_dict
    # get distance top rank
    distance_rank_dict = sorted(distance_rank_dict.items(), key=lambda item: item[1])
    distance_rank_dict.reverse()
    # to see how many ones are sharing the first place
    first_keys, first_key_count = [], distance_rank_dict[0][1]
    for k, v in distance_rank_dict:
        if first_key_count == v:
            first_keys.append(k)
    # if got only one first, then is the prediting label
    if len(first_keys) == 1:
        return first_keys[0], distance_rank_dict
    # otherwise, compute the scores for each class
    key_score_dict = {}
    for key in first_keys:
        for _i in range(top_num):
            k, v = distance_dict[_i]
            if k.split('_')[0] != key:
                continue
            # one drop of the rank, 0.25 drop of the score
            score = top_num - (_i * 0.25)
            if key not in key_score_dict:
                key_score_dict[key] = score
            else:
                key_score_dict[key] += score
    # make the class that wins the prediting label
    max_score_key = max(key_score_dict.items(), key=lambda item: item[1])[0]
    return max_score_key, key_score_dict


def t_knn(base_model, lst, sample_num_each_cls=5, margin=5, epoch=1, test_dir='datas/dishes_dataset/test_std', feature_dir='evaluate_result/feature_map'):  # test
    feature_map_name = 'margin({})_epoch({})_featureMap_{}_{}.pkl'.format(margin, epoch, len(lst), sample_num_each_cls)
    feature_map = pickle_read(os.path.join(feature_dir, feature_map_name))
    positive_num, negative_num, loop_count = 0, 0, 0

    start_time = time.time()
    test_files = os.listdir(test_dir)

    for test_file in test_files: # in this loop we test all images one by one
        file_path = os.path.join(test_dir, test_file)
        ground_truth_label = re.split('_', file_path)[-1][:-4]  # accroding to the directory name

        if int(ground_truth_label) not in lst:  # ugly code except the class we do not need in test
            continue
        prediction, _result = evaluate_single_file_knn(file_path, feature_map, base_model, ground_truth_label=ground_truth_label, base_num=sample_num_each_cls)
        loop_count += 1
        if prediction == ground_truth_label:
            positive_num += 1
        else:
            negative_num += 1
        if loop_count % 400 == 0:
            print('all:', loop_count, ', positive:', positive_num, ', negative:', negative_num)
    print('this evaluation take time:', time.time() - start_time)  # the time we use in the test
    return positive_num / len(test_files)


def t_with_dish_knn(base_model, lst, sample_num_each_cls=5, margin=5, epoch=1, test_dir='datas/dishes_dataset/test_std', feature_dir='evaluate_result/feature_map'):  # test
    feature_map_name = 'margin({})_epoch({})_featureMap_{}_{}.pkl'.format(margin, epoch, len(lst), sample_num_each_cls)
    feature_map = pickle_read(os.path.join(feature_dir, feature_map_name))
    positive_num, negative_num, loop_count = 0, 0, 0
    undecision_num = 0

    start_time = time.time()
    test_files = os.listdir(test_dir)

    for test_file in test_files: # in this loop we test all images one by one
        file_path = os.path.join(test_dir, test_file)
        ground_truth_label = re.split('_', file_path)[-1][:-4]  # accroding to the directory name

        if int(ground_truth_label) not in lst:  # ugly code except the class we do not need in test
            continue

        prediction_list, _ = evaluate_single_file_knn(file_path, feature_map, base_model, ground_truth_label=ground_truth_label, base_num=sample_num_each_cls)

        predict_dish_type_list = dish_map_dict[ground_truth_label]  # 100 %
        is_decision = False
        predict_label = ''

        if type(prediction_list) == list:
            for (k, v) in prediction_list:
                if is_decision == True:
                    break
                _predict_dish_type_list = dish_map_dict[k]

                for _dish_type in _predict_dish_type_list:
                    if _dish_type in predict_dish_type_list:
                        predict_label = k
                        is_decision = True
                        break
        else:
            for (k, v) in prediction_list.items(): # it's not list but dict
                if is_decision == True:
                    break
                _predict_dish_type_list = dish_map_dict[k]

                for _dish_type in _predict_dish_type_list:
                    if _dish_type in predict_dish_type_list:
                        predict_label = k
                        is_decision = True
                        break


        if is_decision == False:
            undecision_num += 1
        elif is_decision == True and predict_label == ground_truth_label:
            positive_num += 1
        elif is_decision == True and predict_label != ground_truth_label:
            negative_num += 1

        loop_count += 1
        if loop_count % 400 == 0:
            print('all:{}, positive:{}, negative:{}, undecision:{}'.format(loop_count, positive_num, negative_num, undecision_num))
            # print('all:', loop_count, ', positive:', positive_num, ', negative:', negative_num)

    print('this evaluation take time:', time.time() - start_time)  # the time we use in the test
    return positive_num / len(test_files)


def t_with_dish_minimum_average_distance(base_model, lst, sample_num_each_cls=5, margin=5, epoch=1, test_dir='datas/dishes_dataset/test_std', feature_dir='evaluate_result/feature_map'):  # test
    feature_map_name = 'margin({})_epoch({})_featureMap_{}_{}.pkl'.format(margin, epoch, len(lst), sample_num_each_cls)
    feature_map = pickle_read(os.path.join(feature_dir, feature_map_name))
    test_file_name_list = os.listdir(test_dir)

    rank_map = dict()
    num_map = dict()
    positive_num = dict()
    first_num = dict()
    rate_dict = dict()

    for i in lst:
        first_num[str(i)] = dict()
        positive_num[str(i)] = 0
        num_map[str(i)] = 0
        for j in lst:
            first_num[str(i)][str(j)] = 0
    for i in test_file_name_list:
        file_path = os.path.join(test_dir, i)
        cls_idx = re.split('_', file_path)[-1][:-4]  # accroding to the directory name

        if int(cls_idx) not in lst:  # ugly code except the class we do not need in test
            continue
        tmp_dict, rank_list = evaluate_single_file_with_average_feature_map(file_path, feature_map, base_model)
        real_predict_label = None
        is_predict = False
        for predict_label in rank_list:
            if is_predict == True:
                break
            for _dish_type in dish_map_dict[predict_label]:
                if _dish_type in dish_map_dict[str(cls_idx)]:
                    real_predict_label = predict_label
                    is_predict = True
                    break
        if real_predict_label == cls_idx:
            positive_num[cls_idx] += 1
        num_map[cls_idx] += 1  # compute all test num of the class
    for k, v in positive_num.items():
        rate_dict[k] = v / (num_map[k] + 1e-12)  # to avoid 0
    return rate_dict, rank_map, positive_num, num_map


def t_save_file(feature_map, base_model, lst, sample_num_each_cls=5, margin=5, epoch=1, test_dir='./test_data/', feature_dir='evaluate_result/feature_map'):  # test

    t1 = time.time()
    # feature_map_name = 'margin({})_epoch({})_featureMap_{}_{}.pkl'.format(margin, epoch, len(lst), sample_num_each_cls)

    # feature_map = pickle_read(os.path.join(feature_dir, feature_map_name))

    # feature_map = transform_feature_map_to_everage(5, 55)
    # print(feature_map.keys())
    test_file_name_list = os.listdir(test_dir)
    # print('files', len(test_file_name_list))

    rank_map = dict()
    num_map = dict()
    positive_num = dict()
    first_num = dict()
    rate_dict = dict()

    for i in lst:
        first_num[str(i)] = dict()
        positive_num[str(i)] = 0
        num_map[str(i)] = 0

        for j in lst:
            first_num[str(i)][str(j)] = 0


    # j = 0  # no need
    all_count, positive_count = 0, 0

    for i in test_file_name_list:
        file_path = os.path.join(test_dir, i)
        cls_idx = re.split('.', file_path)[1] # accroding to the directory name
        # print('cls_idx:', cls_idx)

        if int(cls_idx) not in lst:  # ugly code except the class we do not need in test
            # print('test dont handle class like:', cls_idx)
            continue
        all_count += 1

        tmp_dict = evaluate_single_file_with_average_feature_map(file_path, feature_map, base_model)
        tmp_dict = tmp_dict[0]
        # print(cls_idx, tmp_dict)
        if tmp_dict[str(cls_idx)] == 0:  # compute the correct num of the class
            positive_num[str(cls_idx)] += 1
            positive_count += 1
        if cls_idx not in rank_map:  # compute the rank
            rank_map[cls_idx] = tmp_dict
        else:
            for k in tmp_dict.keys():
                rank_map[cls_idx][k] += tmp_dict[k]
                if tmp_dict[k] == 0:
                    first_num[cls_idx][k] += 1

        num_map[str(cls_idx)] += 1  # compute all test num of the class

        if all_count % 500 == 0:
            print('now all:', all_count, ', and positive:', positive_count)

    for i in range(1, len(lst) + 1):
        _key = str(i)
        if _key in positive_num.keys():
            _acc = positive_num[_key] / (num_map[_key] + 1e-12)
            # with open('./evaluate_result/acc_for_class.txt', 'ab+') as f:
            #     f.write(('class:{:2}, accuracy:{:.5}\n'.format(_key, _acc)).encode())
            # if _acc < 0.8:
            #     print('class:{}, accuracy:{}'.format(_key, _acc))
            # print('class:{}, accuracy:{}'.format(_key, positive_num[_key]/(num_map[_key]+1e-12)))
        # j += 1

    suffix = '{}_{}'.format(len(lst), sample_num_each_cls)
    prefix = 'margin({})_epoch({})_'.format(margin, epoch)

    t2 = time.time()
    print('time for testing', '%.2f' % (t2 - t1), 's')  # the time we use in the test

    save_path = "evaluate_result/all_result/"

    for k, v in positive_num.items():
        rate_dict[k] = v / (num_map[k] + 1e-12)  # to avoid 0

    for k, v in rank_map.items():
        for cls_idx in v.keys():
            rank_map[k][cls_idx] /= num_map[k]

    # pickle_write(os.path.join(save_path, prefix + 'num_map_' + suffix), num_map)  # all prediction of each class
    # pickle_write(os.path.join(save_path, prefix + 'positive_num_' + suffix), positive_num)  # correct prediction of each class
    # pickle_write(os.path.join(save_path, prefix + 'first_num_' + suffix), first_num)   # rank 1st num of each class
    # pickle_write(os.path.join(save_path, prefix + 'all_map_' + suffix), rank_map)  # average rank

    return rate_dict, rank_map, positive_num, num_map


def pickle_read(file_path):
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except:
        print('pickle read error: not exits {}'.format(file_path))
        return None


def pickle_write(file_path, what_to_write):
    try:
        with open(file_path, 'wb+') as f:
            pickle.dump(what_to_write, f)
    except:
        print('pickle write error: {}'.format(file_path))


def get_accuracy_from_map(positive_num, num_map):
    start_time = time.time()
    if type(positive_num) == str:
        positive_num = pickle_read(positive_num)
    if type(num_map) == str:
        num_map = pickle_read(num_map)
    all_positive_num = 0
    all_num = 0
    for key, value in positive_num.items():
        all_positive_num += positive_num[key]
        all_num += num_map[key]
    print('time for get_accuracy_from_map:', '%.1f' % (time.time() - start_time), 's')
    return all_positive_num / (all_num + 1e-12)

def do_get_feature_and_t(base_model, margin, epoch):
    # lst_all = [i for i in range(1, 96)]
    lst_all = [i for i in range(1, 55)]
    lst = [lst_all]
    for i in lst:
        for j in [5]:  # choose 5, 10 samples as the database
            # get_feature_map_k(base_model, i, j, margin=margin, epoch=epoch)
            feature_map = get_feature_map_average(base_model, i, j, margin=margin, epoch=epoch)
            # feature_map = pickle_read('./evaluate_result/feature_map/margin(20)_epoch(1)_featureMap_54_5')
            _, _, positive_num, num_map = t_save_file(feature_map, base_model, i, j, margin=margin, epoch=epoch)
            _accuracy = get_accuracy_from_map(positive_num, num_map)
            print('Accuracy: {} under {} classes, {} samples/class'.format(_accuracy, len(i), j))
            # with open('./evaluate_result/acc_for_class.txt', 'ab+') as f:
            #     f.write(('* Accuracy: {:.5} under {} classes, {} samples/class\n'.format(_accuracy, len(i), j)).encode())
    return _accuracy


if __name__ == '__main__':
    get_sample_std_file(5) # Do this to get 5 sample pictures for every class

    model_root = './model/pytorch-ckpt/'
    for model_dir in [model_root + x for x in ['time1', 'time2', 'time3']]:
        for model_path in [model_dir + '/' + x for x in os.listdir(model_dir) if '.tar' in x]:
            model_name = model_path.split('/')[-1]

            with open(model_dir + '/result.txt', 'ab+') as f:
                f.write(('model:{}\n'.format(model_name)).encode())

            layers = 18 if 'layers18' in model_path else 50
            model = load_model(model_path=model_path, layers=layers)
            model.eval() # tell that is testing now, no need to BP
            acc = do_get_feature_and_t(model, margin=20, epoch=1)  # get feature map and test the model.

            with open(model_dir + '/result.txt', 'ab+') as f:
                f.write(('accuracy: {}\n\n'.format(acc)).encode())
