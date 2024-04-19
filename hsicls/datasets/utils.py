import spectral
import os
from scipy import misc
from hdf5storage import loadmat
import numpy as np
from sklearn.model_selection import train_test_split
from tabulate import tabulate

__all__ = ['get_dataset_info', 'get_gt_info', 'split_gt', 'padding', 'open_file']

def get_dataset_info(hyperx, label_values=None):
    label_values = list(range(hyperx.num_classes)) if label_values is None else label_values
    header = label_values + ['Total']
    info = [0 for _ in header]
    for _, label in hyperx:
        info[label] += 1
        info[-1] += 1
    return tabulate([header, info], headers='firstrow', tablefmt='fancy_grid')

def get_gt_info(gt, label_values=None, ignored_values=[0]):
    header, info = np.unique(gt, return_counts=True)
    label_values = header if label_values is None else np.array(label_values)
    for label in header:
        if label in ignored_values:
            idx = np.where(header == label)[0][0]
            header = np.delete(header, idx)
            info = np.delete(info, idx)
    header = label_values[header].tolist() + ['Total']
    info = info.tolist() + [np.sum(info)]
    return tabulate([header, info], headers='firstrow', tablefmt='fancy_grid')

def split_gt(gt, train_size, test_size=None, mode='random', **params):
    """ Extract a fixed percentage of samples from an array of labels.

    Args:
        * gt: a 2D array of int labels
        * train_size: [0, 1] float or int
        * test_size: [0, 1] float or int or None
        * mode: random or fixed or disjoint

    Returns:
        train_gt, test_gt: 2D arrays of int labels

    """

    train_gt = np.zeros_like(gt)
    test_gt = np.zeros_like(gt)
    if train_size > 1:
       train_size = int(train_size)

    if mode == 'random':
        indices = np.nonzero(gt)
        X = list(zip(*indices)) # x,y features
        y = gt[indices].ravel() # classes
        train_indices, test_indices = train_test_split(X, train_size=train_size, test_size=test_size, stratify=y, random_state=42)
        train_indices = [list(t) for t in zip(*train_indices)]
        test_indices = [list(t) for t in zip(*test_indices)]
        train_gt[*train_indices] = gt[*train_indices]
        test_gt[*test_indices] = gt[*test_indices]
    elif mode == 'fixed':
        assert type(train_size) == int
        train_ratio = params.get('fixed_train_ratio', 0.8)
        train_indices, test_indices = [], []
        labels, counts = np.unique(gt, return_counts=True)
        for l in labels:
            if l == 0:
                continue
            indices = np.nonzero(gt == l) # ((x1,x2,x3,...), (y1,y2,y3,...))
            X = list(zip(*indices)) # ((x1,y1),(x2,y2),(x3,y3),...)

            if counts[l] < train_size / train_ratio:
                train, test = train_test_split(X, train_size=train_ratio, random_state=42)
            else:
                train, test = train_test_split(X, train_size=train_size, test_size=test_size, random_state=42)
            train_indices += train
            test_indices += test
        train_indices = [list(t) for t in zip(*train_indices)]
        test_indices = [list(t) for t in zip(*test_indices)]
        train_gt[*train_indices] = gt[*train_indices]
        test_gt[*test_indices] = gt[*test_indices]
    elif mode == 'disjoint':
        train_gt = np.copy(gt)
        test_gt = np.copy(gt)
        for l in np.unique(gt):
            mask = gt == l
            for x in range(gt.shape[0]):
                first_half_count = np.count_nonzero(mask[:x, :])
                second_half_count = np.count_nonzero(mask[x:, :])
                try:
                    ratio = first_half_count / (first_half_count + second_half_count)
                    if ratio > 0.9 * train_size:
                        break
                except ZeroDivisionError:
                    continue
            mask[:x, :] = 0
            train_gt[mask] = 0

        test_gt[train_gt > 0] = 0
    else:
        raise ValueError("{} sampling is not implemented yet.".format(mode))
    return train_gt, test_gt

def open_file(dataset):
    _, ext = os.path.splitext(dataset)
    ext = ext.lower()
    if ext == '.mat':
        # Load Matlab array
        return loadmat(dataset)
    elif ext == '.tif' or ext == '.tiff':
        # Load TIFF file
        return misc.imread(dataset)
    elif ext == '.hdr':
        img = spectral.open_image(dataset)
        return img.load()
    else:
        raise ValueError("Unknown file format: {}".format(ext))

def padding(img, gt, patch_size):
    assert img.shape[:2] == gt.shape
    p = patch_size // 2
    img = np.pad(img, ((p, p), (p, p), (0, 0)), mode='symmetric')
    gt = np.pad(gt, ((p, p), (p, p)), mode='constant', constant_values=(0, 0))
    return img, gt
