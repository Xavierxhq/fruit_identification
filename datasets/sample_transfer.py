import os, shutil, pickle
import cv2


if __name__ == "__main__":
    rootdir = "/home/ubuntu/Program/xhq/dataset/fruit_sample/single"
    train_save_dir = "../train_data/"
    test_save_dir = "../test_data/"
    if os.path.exists(train_save_dir):
        shutil.rmtree(train_save_dir)
    os.makedirs(train_save_dir)
    if os.path.exists(test_save_dir):
        shutil.rmtree(test_save_dir)
    os.makedirs(test_save_dir)

    mapping_dict = dict()
    train_num_dict = dict()
    test_num_dict = dict()
    class_list = os.listdir(rootdir)
    each_class_train_num = 300
    each_class_test_num = 0
    train_class_limit = 41

    # should be careful because the index is [0, 53]
    # but the class is [1, 54]
    test_class_list = [i for i in range(0, 20)]

    # prepare training data
    index = 10000
    label = 1 # the class index starts from 1
    processed_pic_list = []

    for cls_name in class_list:  # it will in order by index.
        cls_path = os.path.join(rootdir, cls_name)
        file_list_of_cls = os.listdir(cls_path)

        print(cls_name, 'has', len(file_list_of_cls), 'pictures.')
        continue

        mapping_dict[str(label)] = cls_name
        train_num_dict[str(label)] = 0
        while train_num_dict[str(label)] < each_class_train_num:
            for file_name in file_list_of_cls:

                if file_name in processed_pic_list:
                    continue
                processed_pic_list.append(file_name)

                file_path = os.path.join(cls_path, file_name)
                new_file_name = '%d.%d.png' % (index, label)
                shutil.copy(file_path, os.path.join(cls_path, new_file_name))
                # img = cv2.imread(file_path)
                # cv2.imwrite(os.path.join(train_save_dir, str(index) + '_' + str(label) + '.png'), img)
                index += 1
                train_num_dict[str(label)] += 1
                if train_num_dict[str(label)] == each_class_train_num:
                    break
        label += 1
        if label == train_class_limit:  # beacuse we only train the class in [1, 41)
            break

    index = 10000  # reset

    for cls_idx in test_class_list:
        label = cls_idx + 1
        cls_name = class_list[cls_idx]
        cls_path = os.path.join(rootdir, cls_name)
        file_list_of_cls = os.listdir(cls_path)
        mapping_dict[str(label)] = cls_name
        test_num_dict[str(label)] = 0
        for file_name in file_list_of_cls:

            if file_name in processed_pic_list:
                continue
            processed_pic_list.append(file_name)

            file_path = os.path.join(cls_path, file_name)
            new_file_name = '%d.%d.png' % (index, label)
            shutil.copy(file_path, os.path.join(cls_path, new_file_name))
            # img = cv2.imread(file_path)
            # cv2.imwrite(os.path.join(test_save_dir, str(index) + '_' + str(label) + '.png'), img)
            index += 1
            test_num_dict[str(label)] += 1
            if test_num_dict[str(label)] == each_class_test_num and each_class_test_num != 0:
                break
        label += 1

    save_dir = '../evaluate_result/all_result/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    f = open(os.path.join(save_dir, 'train_mapping_dict.pkl'), 'wb+')
    pickle.dump(mapping_dict, f)
    f.close()

    f = open(os.path.join(save_dir, 'train_num_dict.pkl'), 'wb+')
    pickle.dump(train_num_dict, f)
    f.close()

    f = open(os.path.join(save_dir, 'test_num_dict.pkl'), 'wb+')
    pickle.dump(test_num_dict, f)
    f.close()
