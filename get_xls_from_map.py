import pickle
import os
from collections import OrderedDict
import json
import re
import xlwt


def init_dict(st_range, nd_range, init_value=0):
    the_dict = dict()
    for i in range(st_range):
        the_dict[str(i)] = dict()
        for j in range(nd_range):
            the_dict[str(i)][str(j)] = init_value

    return the_dict


def pickle_read(file_path):
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except:
        print('pickle read error: not exits {}'.format(file_path))
        return None


def get_text_result():
    base_dir = './'

    class_num = ['54']
    sample_num = ['_5', '_10']
    positive_map_str = 'positive_num_'
    num_map_str = 'num_map_'
    all_map_str = 'all_map_'
    first_num_str = 'first_num_'
    for i in class_num:
        for j in sample_num:
            positive_map_name = positive_map_str + str(i) + str(j)
            num_map_name = num_map_str + str(i) + str(j)
            all_map_name = all_map_str + str(i) + str(j)
            first_num_name = first_num_str + str(i) + str(j)
            positive_map_path = os.path.join(base_dir, positive_map_name)
            num_map_path = os.path.join(base_dir, num_map_name)
            all_map_path = os.path.join(base_dir, all_map_name)
            first_num_path = os.path.join(base_dir, first_num_name)
            positive_map = pickle_read(positive_map_path)
            num_map = pickle_read(num_map_path)
            all_map = pickle_read(all_map_path)
            first_num = pickle_read(first_num_path)

            result_dict = OrderedDict()
            write_excel(first_num, 'result_' + str(i) + str(j) + '.xls')
            for cls in range(55):  # read the class index in order to make the result in order.
                cls_idx = str(cls)
                if cls_idx in positive_map:
                    result_dict[cls_idx] = positive_map[cls_idx] / num_map[cls_idx]
                    f = open(i + j + '.txt', 'w+')
                    f.write('accuracy:')
                    f.write(json.dumps(result_dict, indent=4))
                    f.write('average rank of each class')
                    f.write(json.dumps(all_map, indent=4))
                    f.close()


def write_excel(new_all_map, file_name):
    mapping_dict = pickle_read('mapping_dict')
    book = xlwt.Workbook()
    sheet = book.add_sheet('Sheet1', cell_overwrite_ok=True)
    row_index = 1
    col_index = 1
    # print(new_all_map)
    for i in range(1, 55):
        if str(i) in new_all_map.keys():
            sheet.write(i, 0, mapping_dict[str(i)])
            sheet.write(0, i, mapping_dict[str(i)])

    for i in range(55):
        if str(i) in new_all_map.keys():
            # sheet.write(row_index, 1, i)
            row_sum = 0
            for j in range(55):
                # print(new_all_map[str(i)])
                if str(j) in new_all_map[str(i)]:
                    # sheet.write(1, col_index, j)
                    # print(new_all_map[str(i)][str(j)])
                    row_sum += new_all_map[str(i)][str(j)]
                    sheet.write(row_index, col_index, new_all_map[str(i)][str(j)])
                    col_index += 1
            row_correct = new_all_map[str(i)][str(i)]

            if row_sum != 0:
                sheet.write(row_index, col_index, row_correct/row_sum)

            row_index += 1
            col_index = 1
    assert file_name[-4:] == '.xls'
    book.save(file_name)
    pass


def record_margin(file_name='margin_3_epoch_1.txt'):
    f = open(file_name, 'r')
    fr = f.readline()

    dist_dict = init_dict(55, 55)
    num_dict = init_dict(55, 55)
    avg_dict = init_dict(55, 55)
    max_dict = init_dict(55, 55)
    min_dict = init_dict(55, 55, 1000000)

    while fr != '':
        line = re.split(',|:|\n| ', fr)

        cls_1 = line[0]
        cls_2 = line[2]
        dist = line[4]

        num_dict[str(cls_1)][str(cls_2)] += 1
        dist_dict[str(cls_1)][str(cls_2)] += float(dist)
        max_dict[str(int(cls_1) - 1)][str(int(cls_2) - 1)] = max(max_dict[str(int(cls_1) - 1)][str(int(cls_2) - 1)],
                                                                 float(dist))
        min_dict[str(int(cls_1) - 1)][str(int(cls_2) - 1)] = min(min_dict[str(int(cls_1) - 1)][str(int(cls_2) - 1)],
                                                                 float(dist))
        fr = f.readline()
    f.close()

    for i in range(55):
        for j in range(55):
            if num_dict[str(i)][str(j)] == 0:
                continue
            avg_dict[str(i - 1)][str(j - 1)] = dist_dict[str(i)][str(j)] / num_dict[str(i)][str(j)]

            # sheet.write(i, j, dist)
    # book.save('average.xls')
    write_excel(avg_dict, 'average_3_54.xls')
    write_excel(max_dict, 'max_3_54.xls')
    write_excel(min_dict, 'min_3_54.xls')


if __name__ == '__main__':
    record_margin()