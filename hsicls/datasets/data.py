import numpy as np
from torch.utils.data import Dataset

__all__ = ['get_dataset', 'HyperX']

def get_dataset(loader, target_folder="datasets", ignored_values=[0]):
    img, gt, label_values, rgb_bands, palette = loader(target_folder)
    gt = gt.astype(np.int32)

    # Filter NaN out
    nan_mask = np.isnan(img.sum(axis=-1))
    if np.count_nonzero(nan_mask) > 0:
        print(
            "Warning: NaN have been found in the data. It is preferable to remove them beforehand. Learning on NaN data is disabled."
        )
    img[nan_mask] = 0
    gt[nan_mask] = 0

    # Ignore some classes
    label_map = np.arange(len(label_values), dtype=np.int32)
    for l in ignored_values:
        if l == 0:
            continue
        label_map[l] = 0
        label_map[l+1:] -= 1
        del label_values[l]
    gt = label_map[gt]

    # Normalization
    img = np.asarray(img, dtype="float32")
    img = (img - np.min(img)) / (np.max(img) - np.min(img))

    return img, gt, label_values, rgb_bands, palette

class HyperX(Dataset):
    def __init__(self, img, gt, patch_size, transform=None, supervision='full'):
        super(HyperX, self).__init__()
        assert img.shape[:2] == gt.shape, f"Image and GT shapes do not match: {img.shape} and {gt.shape}"
        assert supervision in ['full', 'semi'], f"Supervision type not supported: {supervision}"
        assert patch_size % 2 == 1, f"Patch size must be odd: {patch_size}"

        self.image = img
        self.patch_size = patch_size
        self.transform = transform
        self.num_classes = gt.max()

        mask = np.ones_like(gt)
        if supervision == 'full':
            mask[gt == 0] = 0
            l = -1
        elif supervision == 'semi':
            l = 0

        x_pos, y_pos = np.nonzero(mask)
        p = self.patch_size // 2
        self.indices = [
            (x, y)
            for x, y in zip(x_pos, y_pos)
            if x >= p and x < img.shape[0] - p and y >= p and y < img.shape[1] - p
        ]
        self.labels = [gt[x, y]+l for x, y in self.indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y = self.indices[i]
        x1, y1 = x - self.patch_size // 2, y - self.patch_size // 2
        x2, y2 = x1 + self.patch_size, y1 + self.patch_size

        image = self.image[x1:x2, y1:y2]
        if self.transform:
            image = self.transform(image)
        label = self.labels[i]

        if self.patch_size == 1:
            image = image[:, 0, 0]

        return image, label
