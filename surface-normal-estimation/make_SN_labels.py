#!/usr/bin/env python3

import os
import scipy.io
import sklearn.metrics
import numpy as np
from PIL import Image
import imagesize
import json
from io_utils import load_pickle, save_pickle
import h5py

NUM_NORMALS_CLASSES = 40
IGNORE_LABEL_NORMALS = 255


def assign_normals_to_codebook(normals, codebook):
    assert normals.ndim == 3
    assert codebook.ndim == 2
    assert normals.shape[2] == codebook.shape[1]
    h, w, c = normals.shape[0], normals.shape[1], normals.shape[2]

    flatten_normals = normals.reshape(h * w, c)
    nearest_index, distances = \
        sklearn.metrics.pairwise_distances_argmin_min(flatten_normals, codebook)
    labels = nearest_index.reshape(h, w).astype(np.int32)
    return labels


def make_normal_labels_old(mat_files_dir, dst_lab_files_dir, vocab_file):
    '''
      Creates png files with surface normal labels. Used for training models.
    '''
    # f = h5py.File(vocab_file, 'r')
    vocab = scipy.io.loadmat(vocab_file)
    codebook = vocab['vocabs'][0][0]['normals'][0][0]
    codebook = codebook.astype(np.float64)
    file_list = os.listdir(mat_files_dir)
    file_list = [x for x in file_list if x.endswith('mat') and x.find('nm') == -1]
    file_list.sort()
    assert NUM_NORMALS_CLASSES == codebook.shape[0]
    # whether to create Left-Right Flipped ground truth. This is beta tested only.
    flip = False # flip = True is beta
    for ind, file_name in enumerate(file_list):
        file_path = os.path.join(mat_files_dir, file_name)
        # dt = h5py.File(file_path, 'r')
        dt = scipy.io.loadmat(file_path)
        print(dt)
        raw_normals = [dt['nx'], dt['ny'], dt['nz']]
        valid_depth = dt['depthValid']
        raw_normals = np.stack(raw_normals).astype(np.float64)
        raw_normals = np.transpose(raw_normals, (1, 2, 0))
        normal_labels = assign_normals_to_codebook(raw_normals, codebook)
        assert normal_labels.min() >= 0 and normal_labels.max() < NUM_NORMALS_CLASSES
        # now account for valid_depth mask
        normal_labels[valid_depth == 0] = IGNORE_LABEL_NORMALS
        dst_name = file_name.replace('.mat', '_SN%d.png' % (NUM_NORMALS_CLASSES))
        dst_file_path = os.path.join(dst_lab_files_dir, dst_name)
        im = Image.fromarray(normal_labels.astype(np.uint8))
        im.save(dst_file_path)
        if ind % 100 == 0:
            print('Wrote %s' % (dst_file_path))
        if flip:
            # first we do a horizontal flip
            flipped_normals = raw_normals[:, ::-1, :]

            # now we need to do a rotation
            # flip the "X" axis
            flipped_normals[:, :, 0] *=  -1.

def make_normal_labels(mat_files_dir, dst_lab_files_dir, vocab_file):
    '''
      Creates png files with surface normal labels. Used for training models.
    '''
    # f = h5py.File(vocab_file, 'r')
    vocab = scipy.io.loadmat(vocab_file)
    codebook = vocab['vocabs'][0][0]['normals'][0][0]
    codebook = codebook.astype(np.float64)
    normal_info = load_pickle('/raid/hgouk/ssl-normals/surfacenormal_metadata/all_normals.pklz')
    assert NUM_NORMALS_CLASSES == codebook.shape[0]
    # whether to create Left-Right Flipped ground truth. This is beta tested only.
    flip = False # flip = True is beta
    iterator = zip(normal_info['all_normals'], normal_info['all_valid_depth_masks'], normal_info['all_filenames'])
    for ind, (raw_normals, valid_depth, file_name)  in enumerate(iterator):
        normal_labels = assign_normals_to_codebook(raw_normals, codebook)
        assert normal_labels.min() >= 0 and normal_labels.max() < NUM_NORMALS_CLASSES
        # now account for valid_depth mask
        normal_labels[valid_depth == 0] = IGNORE_LABEL_NORMALS
        dst_name = file_name + '_SN%d.png' % (NUM_NORMALS_CLASSES)
        dst_file_path = os.path.join(dst_lab_files_dir, dst_name)
        im = Image.fromarray(normal_labels.astype(np.uint8))
        im.save(dst_file_path)
        if ind % 100 == 0:
            print('Wrote %s' % (dst_file_path))
        if flip:
            # first we do a horizontal flip
            flipped_normals = raw_normals[:, ::-1, :]

            # now we need to do a rotation
            # flip the "X" axis
            flipped_normals[:, :, 0] *=  -1.


def make_metadata_json_and_restructure_images(image_dir, splits_dir, copy_to_dir, json_dir):
    '''
     Makes metadata json file containing all the image information.
     Copies images to a separate dir in a format suitable for training
     image_dir: where images are stored
     splits_dir: where the text files containing train/test splits are stored
     copy_to_dir: where the images will be copied to
     json_dir: where metadata json will be created
    '''
    for split in ['train', 'test']:
        splits_file = os.path.join(splits_dir, '%slist.txt' % (split))
        json_name = os.path.join(json_dir, split + '_SN%d.json' % (NUM_NORMALS_CLASSES))
        records = []
        with open(splits_file, 'r') as fh:
            for line in fh:
                record = {}
                image_id = int(line.strip())
                src_image_name = line.strip() + '_rgb.jpg'
                src_image_path = os.path.join(image_dir, src_image_name)
                if copy_to_dir is not None:
                    dst_image_name = '%06d_rgb.jpg' % (image_id)
                    dst_image_path = os.path.join(copy_to_dir, dst_image_name)
                    os.system('cp %s %s' % (src_image_path, dst_image_path))
                    image_name = dst_image_name
                    image_path = dst_image_path
                else:
                    image_name = src_image_name
                    image_path = src_image_path
                labels_name = '%06d' % (image_id) + '_SN%d.png' % (NUM_NORMALS_CLASSES)
                record['img'] = image_name
                record['segm'] = labels_name
                im_w, im_h = imagesize.get(image_path)
                record['width'] = im_w
                record['height'] = im_h
                records.append(record)
        with open(json_name, 'w') as fh:
            json.dump(records, fh)
        print('Wrote %s' % (json_name))


def convert_matlab_normals_to_numpy_files(mat_files_dir, dst_dir):
    file_list = os.listdir(mat_files_dir)
    file_list = [x for x in file_list if x.endswith('mat') and x.find('nm') == -1]
    file_list.sort()
    all_valid_depth_masks = []
    all_normals = []
    for _ind, file_name in enumerate(file_list):
        file_path = os.path.join(mat_files_dir, file_name)
        dt = scipy.io.loadmat(file_path)
        raw_normals = [dt['nx'], dt['ny'], dt['nz']]
        raw_normals = np.stack(raw_normals).astype(np.float32)
        valid_depth = dt['depthValid'].astype(np.float32)
        raw_normals = np.transpose(raw_normals, (1, 2, 0))
        all_valid_depth_masks.append(valid_depth)
        all_normals.append(raw_normals)

    # make filenames
    all_filenames = [x.replace('.mat', '') for x in file_list]
    # concat
    all_normals = np.stack(all_normals)
    all_valid_depth_masks = np.stack(all_valid_depth_masks)
    assert all_normals.shape[0] == all_valid_depth_masks.shape[0]
    assert len(all_filenames) == all_normals.shape[0]
    assert all_normals.shape[0] == 1449
    save_dict = {
        'all_normals': all_normals,
        'all_valid_depth_masks': all_valid_depth_masks,
        'all_filenames': all_filenames,
    }
    dst_file = os.path.join(dst_dir, 'all_normals.pklz')
    save_pickle(dst_file, save_dict)
    print('Wrote', dst_file)


def main():
    # The following directories contain data as input for the script
    image_dir = '/raid/hgouk/ssl-normals/all_rgb' # directory containing NYU images
    splits_dir = '/raid/hgouk/ssl-normals/surfacenormal_metadata' # directory containing train/test splits text files
    mat_files_dir = '/raid/hgouk/ssl-normals/surfacenormal_metadata' # directory containing .mat files ground truth

    # The following directories are populated with data
    dst_normals_dir = '/raid/hgouk/ssl-normals/all_normals' # dst dir where normal files will be created
    copy_to_dir = 'tmp' # directory where images are copied in the format needed for training
    json_dir = 'tmp' # directory where metadata json is created
    if not os.path.isdir(dst_normals_dir):
        os.makedirs(dst_normals_dir)

    # If you download the ground truth, the all_normals.pklz file is included in it.
    # If you want to re-create it, uncomment the function below
    # convert_matlab_normals_to_numpy_files(mat_files_dir, dst_normals_dir)
    vocab_file = '/raid/hgouk/ssl-normals/surfacenormal_metadata/vocab%d.mat' % (NUM_NORMALS_CLASSES)
    make_normal_labels(mat_files_dir, dst_normals_dir, vocab_file)
    # make_metadata_json_and_restructure_images(image_dir, splits_dir, copy_to_dir, json_dir)


if __name__ == '__main__':
    main()
