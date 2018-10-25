import os
import pickle


if __name__ == '__main__':
    rootdir = "/home/ubuntu/Program/Tableware/data/2018043000/样本/样本"
    lst = os.listdir(rootdir)
    _size = 0
    for dir_name in lst:
        dir_path = os.path.join(rootdir, dir_name)
        if len(os.listdir(dir_path)) > 300:
            _size += 1
    print(_size)

    the_dir = 'evaluate_result/all_result'
    f = open(os.path.join(the_dir, 'mapping_dict'), 'rb')
    mapping_dict = pickle.load(f)
    print(mapping_dict)
    f.close()


    f = open(os.path.join(the_dir, 'test_num_dict'), 'rb')
    test_num_dict = pickle.load(f)
    print(test_num_dict)
    f.close()