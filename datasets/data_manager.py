import glob
import re
from os import path as osp

"""Dataset classes"""


class Tableware(object):


    def __init__(self, root, **kwargs):
        self.dataset_dir = root
        self.train_dir = self.dataset_dir
        # self.train_dir = osp.join(self.dataset_dir, 'train')
        # self.test_dir = osp.join(self.dataset_dir, 'test_std')

        self._check_before_run()
        # if training set label: {1, 12, 3, 4, 67, 8, 102}, it's necessary to relabel!
        train, num_train_pids, num_train_imgs = self._process_dir(self.train_dir, relabel=True)
        # test, num_test_pids, num_test_imgs = self._process_dir(self.test_dir, relabel=False)

        # num_total_pids = num_train_pids + num_test_pids
        # num_total_imgs = num_train_imgs + num_test_imgs

        print("=> Tableware loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # images")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_imgs))
        # print("  test    | {:5d} | {:8d}".format(num_test_pids, num_test_imgs))
        # print("  ------------------------------")
        # print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_imgs))
        print("  ------------------------------")

        self.train = train
        # self.test = test

        self.num_train_pids = num_train_pids
        # self.num_test_pids = num_test_pids

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.isdir(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.isdir(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        # if not osp.isdir(self.test_dir):
        #     raise RuntimeError("'{}' is not available".format(self.test_dir))

    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.png'))

        # target_classes = [str(i) for i in range(55)]
        # img_paths = [x for x in img_paths if x.split('/')[-1].split('.')[0].split('_')[-1] in target_classes]
        # print('length of img_paths:', len(img_paths))

        pattern = re.compile(r'([\d]+)_([\d]+)')

        pid_container = set()
        for img_path in img_paths:
            _, pid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            _, pid= map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue  # junk images are just ignored
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid))

        num_pids = len(pid_container)
        num_imgs = len(dataset)
        return dataset, num_pids, num_imgs

