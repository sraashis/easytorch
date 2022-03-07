import os as _os
import random as _random
import warnings as _warn
from abc import ABC as _ABC

import numpy as _np
import torchvision.transforms as _tmf
from easytorch.data import ETDataset as _ETDataset
import easytorch.vision.imageutils as _imgutils

sep = _os.sep


class BaseImageDataset(_ETDataset, _ABC):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.transforms = self.get_transforms()
        self.pil_to_tensor = _tmf.Compose([_tmf.ToPILImage(), _tmf.ToTensor()])

    def get_transforms(self):
        return _tmf.Compose([_tmf.ToPILImage(), _tmf.ToTensor()])

    def _validate_image_data(self, dspec, img_obj):
        pass

    def load_img(self, dspec, file):
        img_obj = _imgutils.Image()

        """Load Image"""
        img_obj.load(dspec['data_dir'], file)
        img_obj.apply_clahe()
        if len(img_obj.array.shape) > 2 and img_obj.array.shape[-1] != 3:
            _warn.warn(f"Suspicious Image shape: {img_obj.array.shape}, clipping to RGB: {dspec['name']}")
            img_obj.array = img_obj.array[:, :, :3]

        """Load ground truth"""
        dspec['has_gt'] = any(['label_dir' in dspec.keys()])
        if dspec['has_gt']:
            img_obj.load_ground_truth(dspec["label_dir"], dspec["label_getter"])
        else:
            _warn.warn(f"Random label initialized: {dspec['name']}")
            img_obj.ground_truth = _np.random.randint(
                0,
                self.args.get('num_class', 2), img_obj.array.shape[:2]
            ).astype(_np.uint8) * 255

        """Load mask"""
        dspec['has_mask'] = any(['mask_dir' in dspec.keys()])
        if dspec['has_mask']:
            img_obj.load_mask(dspec['mask_dir'], dspec['mask_getter'])

        self._validate_image_data(dspec, img_obj)
        return img_obj

    def __getitem__(self, index):
        raise NotImplementedError('Must implement')


class PatchedImgDataset(BaseImageDataset, _ABC):
    def load_index(self, dataset_name, file):
        r"""
        :param dataset_name: name of teh dataset as provided in train_dataspecs
        :param file: Name of an image
        :return:
        Logic split an image to patches and feed to U-Net. Meanwhile we need to store the four-corners
            of each patch so that we can rejoin the full image from the patches' corresponding predictions.
        """
        dt = self.dataspecs[dataset_name]
        obj = self.load_img(dt, file)

        cache_key = self.diskcache.add(f"{dataset_name}_{file}", obj)
        for corners in _imgutils.get_chunk_indexes(
                obj.array.shape[:2],
                dt['patch_shape'],
                dt['patch_offset']
        ):
            """
            get_chunk_indexes will return the list of four corners of all patches of the images  
            by using window size of self.patch_shape, and offset  of elf.patch_offset
            """
            self.indices.append([dataset_name, file] + corners + [cache_key])


class BinarySemSegImgPatchDataset(PatchedImgDataset):
    def __init__(self, **kw):
        super().__init__(**kw)

    def _validate_image_data(self, dspec, img_obj):
        thr_manual = dspec.setdefault('thr_manual', 50)
        if dspec.get('has_gt'):
            gt_unique = _np.unique(img_obj.ground_truth)
            if len(img_obj.ground_truth.shape) > 2:
                _warn.warn(
                    f"Ground truth shape suspicious: {img_obj.ground_truth.shape}, "
                    f"using 1st channel onl:  {dspec['name']}"
                )
                img_obj.ground_truth = img_obj.ground_truth[:, :, 0]

            if sum(gt_unique) == 1:
                _warn.warn(f"Ground truth 1 converted to 255")
                img_obj.ground_truth[img_obj.ground_truth == 1] = 255

            if len(gt_unique) != self.args.get('num_class', 2):
                _warn.warn(
                    f"Number of unique ground truth items != {self.args.get('num_class')} in  {dspec['name']}. "
                    f"\nBinarizing ... {gt_unique}: "
                )
                _imgutils.binarize(img_obj.ground_truth, thr_manual)

        if dspec.get('has_mask'):
            mask_unique = _np.unique(img_obj.mask)
            if len(img_obj.mask.shape) > 2:
                _warn.warn(f"Mask shape suspicious: {img_obj.mask.shape}, using 1st channel only: {dspec['name']}")
                img_obj.mask = img_obj.mask[:, :, 0]

            if sum(mask_unique) == 1:
                _warn.warn(f"Mask truth 1 converted to 255")
                img_obj.mask[img_obj.mask == 1] = 255

            if len(mask_unique) != 2:
                _warn.warn(
                    f"Number of unique items in mask: {mask_unique} in {dspec['name']}"
                )

        if dspec.get('bbox_crop'):
            copy = img_obj.copy()
            img_obj.array, img_obj.ground_truth, img_obj.mask = _imgutils.masked_bboxcrop(
                img_obj.array,
                img_obj.ground_truth,
                offset=dspec.setdefault('bbox_crop_offset', 51),
                threshold=dspec.setdefault('bbox_crop_threshold', 5)
            )

            if img_obj.array.shape[0] < dspec['patch_shape'][0] or img_obj.array.shape[1] < dspec['patch_shape'][1]:
                _warn.warn(
                    f"BBOX crop reversing for "
                    f"{dspec['name']}:{img_obj.file}, shape: {img_obj.array.shape}, {copy.array.shape}"

                )
                img_obj = copy.copy()

        if dspec.get('resize'):
            img_obj.array = _imgutils.resize(img_obj.array, dspec['resize'])
            img_obj.ground_truth = _imgutils.resize(img_obj.ground_truth, dspec['resize'])

            if img_obj.mask is not None:
                img_obj.mask = _imgutils.resize(img_obj.mask, dspec['resize'])

        """Must binarize after resize"""
        if self.args['num_class'] == 2 and dspec.get('resize'):
            _imgutils.binarize(img_obj.ground_truth, thr_manual)
            if dspec.get('has_mask'):
                _imgutils.binarize(img_obj.mask, thr_manual)
        return img_obj

    def __getitem__(self, index):
        """
        :param index:
        :return: dict with keys - indices, input, label
            We need indices to get the file name to save the respective predictions.
        """
        dname, file, row_from, row_to, col_from, col_to, cache_key = self.indices[index]

        obj = self.diskcache.get(cache_key)
        img = obj.array
        gt = obj.ground_truth[row_from:row_to, col_from:col_to]

        p, q, r, s, pad = _imgutils.expand_and_mirror_patch(
            img.shape,
            [row_from, row_to, col_from, col_to],
            self.dataspecs[dname]['expand_by']
        )
        if len(img.shape) == 3:
            pad = [*pad, (0, 0)]

        img = _np.pad(img[p:q, r:s], pad, 'reflect')
        if self.mode == 'train' and _random.uniform(0, 1) <= 0.5:
            img = _np.flip(img, 0)
            gt = _np.flip(gt, 0)

        if self.mode == 'train' and _random.uniform(0, 1) <= 0.5:
            img = _np.flip(img, 1)
            gt = _np.flip(gt, 1)

        img = self.transforms(img)
        gt = self.pil_to_tensor(gt)
        return {'indices': self.indices[index], 'input': img, 'label': gt.squeeze()}


class FullImgDataset(BaseImageDataset):
    def __init__(self, **kw):
        r"""
        Initialize necessary shapes for unet.
        """
        super().__init__(**kw)
        self.labels = None

    def get_transforms(self):
        if self.mode == "train":
            return _tmf.Compose(
                [_tmf.ToPILImage(),
                 _tmf.RandomHorizontalFlip(),
                 _tmf.RandomVerticalFlip(),
                 _tmf.ToTensor()]
            )
        return self.pil_to_tensor

    def _load_labels(self, dspec):
        return None

    def _get_label(self, file):
        raise NotImplementedError('Use file and self.labels to return a correct label')

    def _validate_image_data(self, dspec, img_obj):

        if dspec.get('bbox_crop'):
            copy = img_obj.copy()
            img_obj.array, img_obj.mask = _imgutils.masked_bboxcrop(
                img_obj.array,
                offset=dspec.setdefault('bbox_crop_offset', 31),
                threshold=dspec.setdefault('bbox_crop_threshold', 5)
            )

            if dspec.get('resize') and (
                    img_obj.array.shape[0] < dspec['resize'][0] or img_obj.array.shape[1] < dspec['resize'][1]
            ):
                _warn.warn(f"BBOX crop reversing for {dspec['name']}:{img_obj.file}, shape: {img_obj.array.shape}")
                img_obj = copy.copy()

        if dspec.get('resize'):
            img_obj.array = _imgutils.resize(img_obj.array, dspec['resize'])

            if img_obj.mask is not None:
                img_obj.mask = _imgutils.resize(img_obj.mask, dspec['resize'])

        return img_obj

    def load_img(self, dspec, file):
        img_obj = _imgutils.Image()

        """Load Image"""
        img_obj.load(dspec['data_dir'], file)
        img_obj.apply_clahe()
        if len(img_obj.array.shape) > 2 and img_obj.array.shape[-1] != 3:
            _warn.warn(f"Suspicious Image shape: {img_obj.array.shape}, clipping to RGB: {dspec['name']}")
            img_obj.array = img_obj.array[:, :, :3]

        """Load mask"""
        dspec['has_mask'] = any(['mask_dir' in dspec.keys()])
        if dspec['has_mask']:
            img_obj.load_mask(dspec['mask_dir'], dspec['mask_getter'])

        self._validate_image_data(dspec, img_obj)
        return img_obj

    def load_index(self, dataset_name, file):
        dspec = self.dataspecs[dataset_name]
        if self.labels is None:
            self.labels = self._load_labels(dspec)

        img_obj = self.load_img(dspec, file)
        cache_key = self.diskcache.add(f"{dataset_name}_{file}", img_obj)
        self.indices.append([dataset_name, file, self._get_label(file), cache_key])

    def __getitem__(self, index):
        _dname, file, label, cache_key = self.indices[index]
        obj = self.diskcache.get(cache_key)
        img = obj.array

        img = self.transforms(img)
        return {'indices': self.indices[index], 'input': img, 'label': _np.array(label)}
