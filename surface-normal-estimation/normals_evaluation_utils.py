#!/usr/bin/env python3

# this is based on https://github.com/aayushbansal/MarrRevisited/blob/master/normals/eval/eval_pred_sn.m # noqa
# ALL evaluaton must be done in float64
# otherwise you get underflow errors and wrong results

import numpy as np
import scipy.io
from io_utils import load_pickle
import sklearn
from collections import OrderedDict

# constants
eps64 = np.finfo('float64').eps


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg


class NormalsConverter(object):
    def __init__(self, vocab_file):
        vocab = scipy.io.loadmat(vocab_file)
        codebook = vocab['vocabs'][0][0]['normals'][0][0]
        self.codebook = codebook.astype(np.float64)
        self.normals_dim = codebook.shape[1]

    def convert_discrete_to_continuous(self, per_pixel_probs):
        assert per_pixel_probs.ndim in [2, 3]

        h, w = None, None
        if per_pixel_probs.ndim == 3:
            # flatten it
            h, w, c = \
                per_pixel_probs.shape[0], per_pixel_probs.shape[1],\
                per_pixel_probs.shape[2]
            per_pixel_probs = per_pixel_probs.reshape(h * w, c)
        self.codebook.shape[0] == per_pixel_probs.shape[1]
        self.codebook = self.codebook.astype(np.float64)
        per_pixel_probs = per_pixel_probs.astype(np.float64)

        # matrix multiply each prob with codebook and sum
        converted_normals = np.dot(per_pixel_probs, self.codebook)
        # normalize to unit length
        converted_normals = NormalsConverter.normalize_rows_matrix(converted_normals)
        converted_normals = converted_normals.astype(per_pixel_probs.dtype)

        if h is not None:
            converted_normals = converted_normals.reshape(h, w, self.normals_dim)

        return converted_normals

    def convert_continuous_to_discrete(self, normals):
        assert normals.ndim == 3
        assert self.codebook.ndim == 2
        assert normals.shape[2] == self.codebook.shape[1]
        h, w, c = normals.shape[0], normals.shape[1], normals.shape[2]

        flatten_normals = normals.reshape(h * w, c)
        nearest_index, distances = \
            sklearn.metrics.pairwise_distances_argmin_min(flatten_normals, self.codebook)
        labels = nearest_index.reshape(h, w).astype(np.int32)
        return labels

    @staticmethod
    def normalize_rows_matrix(M):
        assert M.ndim == 2
        per_row_norm = np.linalg.norm(M, axis=1) + eps64
        M_norm = M / per_row_norm[:, None]
        return M_norm

    @staticmethod
    def ascod(M):
        '''
        gives acos in degrees
        '''
        radians = np.arccos(M)
        degrees = np.degrees(radians)
        return degrees

    @staticmethod
    def stable_mean(arr):
        arr = arr.astype(np.float64)
        mean = np.float64(0.0)
        n = np.float64(1)
        for x in arr:
            mean += (x - mean) / n
            n += 1


class NormalsEvaluator(NormalsConverter):
    def __init__(self, vocab_file, gt_file=None):
        super(NormalsEvaluator, self).__init__(vocab_file)
        if gt_file is not None:
            self.gt_data = load_pickle(gt_file)
        self.metrics = None

    def evaluate_probs(self, per_pixel_probs, filenames):
        normals = self.convert_discrete_to_continuous(per_pixel_probs)
        self.evaluate_normals(normals, filenames)

    def evaluate_normals(self, normals, filenames):
        # first collect gt data according to filenames
        relevant_inds = np.zeros(len(filenames), dtype=np.int32)
        for filename in filenames:
            idx = self.gt_data['all_filenames'].index(filename)
            relevant_inds.append(idx)
        gt_normals = self.gt_data['all_normals'][relevant_inds, :]
        valid_mask = self.gt_data['all_valid_depth_masks'][relevant_inds, :]
        return self.compute_normals_metrics(normals, gt_normals, valid_mask)

    def update_normals_metrics(self, normals, gt_normals, valid_mask):
        metrics = self.compute_normals_metrics(normals, gt_normals, valid_mask)
        # now average
        if self.metrics is None:
            self.initialize_average_meters(metrics)
        else:
            self.update_average_meters(metrics)

    def update_average_meters(self, metrics):
        for key in metrics:
            val = metrics[key]
            if isinstance(val, list):
                for v in val:
                    self.metrics[key].update(v)
            else:
                self.metrics[key].update(val)

    def initialize_average_meters(self, metrics):
        self.metrics = OrderedDict()
        for key in metrics:
            val = metrics[key]
            if isinstance(val, list):
                self.metrics[key] = []
                for v in val:
                    v_avg = AverageMeter()
                    v_avg.update(v)
                    self.metrics[key].append(v_avg)
            else:
                v_avg = AverageMeter()
                v_avg.update(val)
                self.metrics[key] = v_avg

    def compute_normals_metrics_from_network_probs(self, probs, gt_normals,
                                                   valid_mask):
        if isinstance(probs, list):
            probs = np.array(probs)
        if isinstance(gt_normals, list):
            gt_normals = np.array(gt_normals)
        if isinstance(valid_mask, list):
            valid_mask = np.array(valid_mask)
        assert probs.ndim == 4
        assert gt_normals.ndim == 4
        assert valid_mask is None or valid_mask.ndim == 3
        assert probs.shape[1] == self.codebook.shape[0]
        probs_image = np.transpose(probs, (0, 2, 3, 1))
        print('Converting discrete normal probs to raw normals')
        normals = []
        for i in range(probs_image.shape[0]):
            normals.append(self.convert_discrete_to_continuous(probs_image[i, ...]))
        normals = np.stack(normals).astype(np.float64)
        return self.compute_normals_metrics(normals,
                                            gt_normals,
                                            valid_mask)

    def compute_normals_metrics(self, normals, gt_normals, valid_mask):
        assert gt_normals.ndim == 4
        assert valid_mask.ndim == 3
        assert normals.shape == gt_normals.shape
        print('Evaluating normal metrics for %d images' % (normals.shape[0]))

        # make everything float64
        normals = normals.astype(np.float64)
        gt_normals = gt_normals.astype(np.float64)

        # flatten
        normals_flatten = normals.reshape(-1, self.normals_dim)
        gt_normals_flatten = gt_normals.reshape(-1, self.normals_dim)

        # normalize again to be sure
        normals_flatten = NormalsConverter.normalize_rows_matrix(normals_flatten)
        gt_normals_flatten = NormalsConverter.normalize_rows_matrix(gt_normals_flatten)
        if valid_mask is not None:
            assert valid_mask.shape == normals.shape[:3]
            valid_mask = valid_mask.astype(np.int32)
            valid_mask_flatten = valid_mask.flatten()
            valid_inds = valid_mask_flatten > 0
            normals_flatten = normals_flatten[valid_inds]
            gt_normals_flatten = gt_normals_flatten[valid_inds]

        # compute dot product
        dp = np.multiply(normals_flatten, gt_normals_flatten)
        dp = dp.sum(axis=1)
        dp_clipped = np.minimum(1, np.maximum(-1, dp))
        dp_valid = dp_clipped

        # now come the metrics
        dp_valid = dp_valid.astype(np.float64)
        dp_valid_angle = NormalsConverter.ascod(dp_valid)
        mean_e = dp_valid_angle.mean()
        median_e = np.percentile(dp_valid_angle, 50)
        angle_thresh = [11.25, 22.5, 30.]
        mean_per_angle = []
        for angle in angle_thresh:
            mval = (dp_valid_angle < angle).mean() * 100
            mean_per_angle.append(float(mval))

        metrics = (
            ('mean', float(mean_e)),
            ('median_e', float(median_e)),
            ('mean_per_angle', mean_per_angle),
            ('angle_thresh', angle_thresh),
        )
        metrics = OrderedDict(metrics)
        return metrics

    def get_metrics(self):
        return self.metrics

    def get_average_metrics(self):
        metrics = self.metrics
        avg_metrics = OrderedDict()
        for key in metrics:
            if isinstance(metrics[key], list):
                avg_metrics[key] = []
                for v in metrics[key]:
                    avg_metrics[key].append(v.average())
            else:
                avg_metrics[key] = self.metrics[key].average()
        return avg_metrics

    def print_metrics(self, metrics=None):
        if metrics is None:
            metrics = self.get_average_metrics()
        return NormalsEvaluator.print_metrics_standalone(metrics)


    @staticmethod
    def print_metrics_standalone(metrics):
        header = ''
        st = ''
        for name in metrics:
            if name == 'angle_thresh':
                continue
            header += name + ' '
            if isinstance(metrics[name], list):
                st += ', '.join(['%.2f' % (x) for x in metrics[name]])
            else:
                st = st + '%.2f, ' % (metrics[name])
        print(header)
        print(st)
