import os
import json
import torch
import lib.utils.data as torchdata
import cv2
from torchvision import transforms
from scipy.misc import imread, imresize
import numpy as np
from io_utils import load_pickle

# Round x to the nearest multiple of p and x' >= x
def round2nearest_multiple(x, p):
    return ((x - 1) // p + 1) * p


NYU_ROOT_DIR="/raid/hgouk/ssl-normals"

CANDIDATE_PATHS_DICT = {
    'ade': [
        './data/ADE20k/ADEChallengeData2016/all_images',
    ],

    'nyuv2sn40': [
        '/raid/hgouk/ssl-normals/all_images',
    ],
}

SPLIT_LIST_FOLDER_DICT = {
    'ade': './data/ADE20k/',
    'nyuv2sn40': os.path.join(NYU_ROOT_DIR, 'surfacenormal_metadata')
}

# these are only needed for evaluation
NYU_NORMALS_FILE = os.path.join(NYU_ROOT_DIR, 'surfacenormal_metadata/all_normals.pklz')
NYU_VOCAB_FILE = os.path.join(NYU_ROOT_DIR, 'surfacenormal_metadata/vocab%d.mat')

class PILToTensorTransform(object):
    """
    Convert PIL Image to Tensor. Does not rescale unlike torchvision ToTensor()
    """

    def __init__(self):
        pass

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be scaled.
        Returns:
            PIL Image: Rescaled image.
        """
        img_array = np.array(img)
        img_tensor = torch.from_numpy(img_array).float()
        return img_tensor

    def __repr__(self):
        return self.__class__.__name__


class Dataset(torchdata.Dataset):
    def __init__(self, opt, split_name, max_sample=-1, batch_per_gpu=1):
        self.dataset_name = opt.dataset
        self.split = split_name
        self.imgSize = opt.imgSize
        self.imgMaxSize = opt.imgMaxSize

        self.random_flip = False
        if hasattr(opt, 'random_flip'):
            self.random_flip = opt.random_flip
        if self.dataset_name == 'nyuv2sn40':
            assert self.random_flip is False
            self.num_classes = 40
            self.normals_file = NYU_NORMALS_FILE
            self.vocab_file = NYU_VOCAB_FILE % self.num_classes
            image_list_file = \
                os.path.join(SPLIT_LIST_FOLDER_DICT[self.dataset_name],
                             self.split + '_SN%d.json' % (self.num_classes))
        elif self.dataset_name == 'nyuv2sn20':
            assert self.random_flip is False
            self.num_classes = 20
            self.normals_file = NYU_NORMALS_FILE
            self.vocab_file = NYU_VOCAB_FILE % self.num_classes
            image_list_file = \
                os.path.join(SPLIT_LIST_FOLDER_DICT[self.dataset_name],
                             self.split + '_SN%d.json' % (self.num_classes))
        elif self.dataset_name == 'ade':
            self.num_classes = 150
            image_list_file = \
                os.path.join(SPLIT_LIST_FOLDER_DICT[self.dataset_name], self.split + '.json')
        # max down sampling rate of network to avoid rounding during conv or pooling
        self.padding_constant = opt.padding_constant
        # down sampling rate of segm labe
        if hasattr(opt, 'segm_downsampling_rate'):
            self.segm_downsampling_rate = opt.segm_downsampling_rate
        self.batch_per_gpu = batch_per_gpu

        # classify images into two classes: 1. h > w and 2. h <= w
        self.batch_record_list = [[], []]

        # override dataset length when trainig with batch_per_gpu > 1
        self.cur_idx = 0
        self.root_dataset = None
        for path in CANDIDATE_PATHS_DICT[self.dataset_name]:
            if os.path.isdir(path):
                self.root_dataset = path
                break

        self.list_sample = json.load(open(image_list_file, 'r'))

        # mean and std
        self.image_mode = opt.image_mode
        assert self.image_mode in ['bgr', 'rgb', 'lab', 'rgb_nonorm']
        if self.image_mode == 'bgr':
            # values for the MIT CSAIL models
            self.img_transform = transforms.Compose([
                transforms.Normalize(mean=[102.9801, 115.9465, 122.7717], std=[1., 1., 1.])
                ])
        elif self.image_mode == 'rgb':
            # values for the torchvision models. Also valid for the paper
            self.img_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        elif self.image_mode == 'rgb_nonorm':
            self.img_transform = transforms.Compose([
                transforms.ToTensor()
            ])
        elif self.image_mode == 'lab':
            self.img_transform = None
        self.if_shuffled = False
        if max_sample > 0:
            self.list_sample = self.list_sample[0:max_sample]
        self.num_sample = len(self.list_sample)
        assert self.num_sample > 0
        print('# samples: {}'.format(self.num_sample))

    def _get_sub_batch(self):
        while True:
            # get a sample record
            this_sample = self.list_sample[self.cur_idx]
            if this_sample['height'] > this_sample['width']:
                self.batch_record_list[0].append(this_sample) # h > w, go to 1st class
            else:
                self.batch_record_list[1].append(this_sample) # h <= w, go to 2nd class

            # update current sample pointer
            self.cur_idx += 1
            if self.cur_idx >= self.num_sample:
                self.cur_idx = 0
                np.random.shuffle(self.list_sample)

            if len(self.batch_record_list[0]) == self.batch_per_gpu:
                batch_records = self.batch_record_list[0]
                self.batch_record_list[0] = []
                break
            elif len(self.batch_record_list[1]) == self.batch_per_gpu:
                batch_records = self.batch_record_list[1]
                self.batch_record_list[1] = []
                break
        return batch_records

    def convertbgr2lab(self, img):
        # img is [0, 255] , HWC, BGR format, uint8 type
        assert img.dtype == np.uint8, 'cv2 expects a uint8 image'
        assert len(img.shape) == 3, 'Image should have dim H x W x 3'
        assert img.shape[2] == 3, 'Image should have dim H x W x 3'
        img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        # 8-bit image range -> L [0, 255], A [0, 255], B [0, 255]. Rescale it to:
        # L [-50, 50], A [-128, 127], B [-128, 127]
        img_lab = img_lab.astype(np.float32)
        img_lab[:, :, 0] = (img_lab[:, :, 0] * (100.0 / 255.0)) - 50.0
        img_lab[:, :, 1:] = img_lab[:, :, 1:] - 128.0
        return img_lab

    def _process_image(self, img):
        if self.image_mode == 'bgr':
            img = img.astype(np.float32)[:, :, ::-1] # RGB to BGR!!!
            img = img.transpose((2, 0, 1))
            img = self.img_transform(torch.from_numpy(img.copy()))
        elif self.image_mode == 'rgb' or self.image_mode == 'rgb_nonorm':
            img = self.img_transform(img)
        elif self.image_mode == 'lab':
            # first convert to BGR
            img_bgr = img[:, :, ::-1] # RGB to BGR!!!
            img_lab = self.convertbgr2lab(img_bgr.astype(np.uint8))
            # now convert to C X H x W
            img_lab = img_lab.transpose((2, 0, 1))
            img = torch.from_numpy(img_lab).float()
        return img

    def __getitem__(self, index):
        if self.split == 'train':
            return self._get_item_train(index)
        elif self.split in ['val', 'test']:
            return self._get_item_test(index)

    def _get_item_train(self, index):
        # NOTE: random shuffle for the first time. shuffle in __init__ is useless
        if not self.if_shuffled:
            np.random.shuffle(self.list_sample)
            self.if_shuffled = True

        # get sub-batch candidates
        batch_records = self._get_sub_batch()

        # resize all images' short edges to the chosen size
        if isinstance(self.imgSize, list):
            this_short_size = np.random.choice(self.imgSize)
        else:
            this_short_size = self.imgSize

        # calculate the BATCH's height and width
        # since we concat more than one samples, the batch's h and w shall be larger than EACH sample
        batch_resized_size = np.zeros((self.batch_per_gpu, 2), np.int32)
        for i in range(self.batch_per_gpu):
            img_height, img_width = batch_records[i]['height'], batch_records[i]['width']
            this_scale = min(this_short_size / min(img_height, img_width), \
                    self.imgMaxSize / max(img_height, img_width))
            img_resized_height, img_resized_width = img_height * this_scale, img_width * this_scale
            batch_resized_size[i, :] = img_resized_height, img_resized_width
        batch_resized_height = np.max(batch_resized_size[:, 0])
        batch_resized_width = np.max(batch_resized_size[:, 1])

        # Here we must pad both input image and segmentation map to size h' and w' so that p | h' and p | w'
        batch_resized_height = int(round2nearest_multiple(batch_resized_height, self.padding_constant))
        batch_resized_width = int(round2nearest_multiple(batch_resized_width, self.padding_constant))

        assert self.padding_constant >= self.segm_downsampling_rate,\
                'padding constant must be equal or large than segm downsamping rate'
        batch_images = torch.zeros(self.batch_per_gpu, 3, batch_resized_height, batch_resized_width)
        batch_segms = torch.zeros(self.batch_per_gpu, batch_resized_height // self.segm_downsampling_rate, \
                                batch_resized_width // self.segm_downsampling_rate).long()

        for i in range(self.batch_per_gpu):
            this_record = batch_records[i]

            # load image and label
            image_path = os.path.join(self.root_dataset, this_record['img'])
            segm_path = os.path.join(self.root_dataset, this_record['segm'])
            img = imread(image_path, mode='RGB')
            segm = imread(segm_path)

            assert(img.ndim == 3)
            assert(segm.ndim == 2)
            assert(img.shape[0] == segm.shape[0])
            assert(img.shape[1] == segm.shape[1])

            if self.random_flip is True:
                random_flip = np.random.choice([0, 1])
                if random_flip == 1:
                    img = cv2.flip(img, 1)
                    segm = cv2.flip(segm, 1)

            # note that each sample within a mini batch has different scale param
            img = imresize(img, (batch_resized_size[i, 0], batch_resized_size[i, 1]), interp='bilinear')
            segm = imresize(segm, (batch_resized_size[i, 0], batch_resized_size[i, 1]), interp='nearest')

            # to avoid seg label misalignment
            segm_rounded_height = round2nearest_multiple(segm.shape[0], self.segm_downsampling_rate)
            segm_rounded_width = round2nearest_multiple(segm.shape[1], self.segm_downsampling_rate)
            segm_rounded = np.zeros((segm_rounded_height, segm_rounded_width), dtype='uint8')
            segm_rounded[:segm.shape[0], :segm.shape[1]] = segm

            segm = imresize(segm_rounded, (segm_rounded.shape[0] // self.segm_downsampling_rate, \
                                           segm_rounded.shape[1] // self.segm_downsampling_rate), \
                            interp='nearest')
             # image to float
            img = self._process_image(img)

            batch_images[i][:, :img.shape[1], :img.shape[2]] = img
            batch_segms[i][:segm.shape[0], :segm.shape[1]] = torch.from_numpy(segm.astype(np.int)).long()

        if self.dataset_name == 'ade':
            batch_segms = batch_segms - 1 # label from -1 to 149
        elif self.dataset_name.startswith('nyuv2'):
            # ignore label is 255 in the png file
            # but the code takes ignore label as -1
            ignore_idxs = batch_segms == 255
            batch_segms[ignore_idxs] = -1
        output = {}
        output['img_data'] = batch_images
        output['seg_label'] = batch_segms
        return output

    def _get_item_test(self, index):
        this_record = self.list_sample[index]
        # load image and label
        image_path = os.path.join(self.root_dataset, this_record['img'])
        segm_path = os.path.join(self.root_dataset, this_record['segm'])
        img = imread(image_path, mode='RGB')
        segm = imread(segm_path)

        ori_height, ori_width, _ = img.shape

        img_resized_list = []
        for this_short_size in self.imgSize:
            # calculate target height and width
            scale = min(this_short_size / float(min(ori_height, ori_width)),
                    self.imgMaxSize / float(max(ori_height, ori_width)))
            target_height, target_width = int(ori_height * scale), int(ori_width * scale)

            # to avoid rounding in network
            target_height = round2nearest_multiple(target_height, self.padding_constant)
            target_width = round2nearest_multiple(target_width, self.padding_constant)

            # resize
            img_resized = cv2.resize(img.copy(), (target_width, target_height))

            # image to float
            img_resized = self._process_image(img_resized)

            img_resized = torch.unsqueeze(img_resized, 0)
            img_resized_list.append(img_resized)

        segm = torch.from_numpy(segm.astype(np.int)).long()

        batch_segms = torch.unsqueeze(segm, 0)

        if self.dataset_name == 'ade':
            batch_segms = batch_segms - 1 # label from -1 to 149
        elif self.dataset_name.startswith('nyuv2'):
            # ignore label is 255 in the png file
            # but the code takes ignore label as -1
            ignore_idxs = batch_segms == 255
            batch_segms[ignore_idxs] = -1
        output = {}
        output['img_ori'] = img.copy()
        output['img_data'] = [x.contiguous() for x in img_resized_list]
        output['seg_label'] = batch_segms.contiguous()
        if self.dataset_name.startswith('nyuv2'):
            output['info'] = this_record['img'].split('_')[0]
        else:
            output['info'] = this_record['img']
        return output

    def __len__(self):
        if self.split == 'train':
            return int(1e10) # It's a fake length due to the trick that every loader maintains its own list
        else:
            return self.num_sample

    def load_normals_file(self):
        normals_file = self.normals_file
        normals_data = load_pickle(normals_file)
        filename_to_id = \
            {normals_data['all_filenames'][x]: x for x in range(len(normals_data['all_filenames']))}

        def get_normal_and_valid_depth(filename):
            id = filename_to_id[filename]
            return normals_data['all_normals'][id], normals_data['all_valid_depth_masks'][id]

        return get_normal_and_valid_depth
