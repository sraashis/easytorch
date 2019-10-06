import os
import traceback
from itertools import islice

sep = os.sep
import numpy as np
import pydicom
from PIL import Image

import img_utils as iu
import random
import cv2


def load_indices(mapping_file, shuffle_indices=False, limit=10e10):
    indices = []
    with open(mapping_file) as infile:
        linecount, six_rows, _ = 1, True, next(infile)
        while six_rows and len(indices) < limit:
            try:

                print('Reading Line: {}'.format(linecount), end='\r')

                six_rows = list(r.rstrip().split(',') for r in islice(infile, 6))
                image_file, cat_label = None, []
                for hname, label in six_rows:
                    (ID, file_ID, htype), label = hname.split('_'), float(label)
                    image_file = ID + '_' + file_ID + '.dcm'
                    cat_label.append(label)

                if image_file and len(cat_label) == 6:
                    indices.append([image_file, np.array(cat_label)])

                linecount += 6
            except Exception as e:
                traceback.print_exc()
    if shuffle_indices:
        random.shuffle(indices)
    return indices


def _save(ix, file, label, images_dir, out_dir, resize_shape):
    print('Writing:{} {}'.format(file, ix), end='\r')
    try:
        dcm = pydicom.dcmread(images_dir + os.sep + file)
        image = dcm.pixel_array.astype(np.int16)

        # Set outside-of-scan pixels to 1
        # The intercept is usually -1024, so air is approximately 0
        image[image == -2000] = 0

        intercept = dcm.RescaleIntercept
        slope = dcm.RescaleSlope
        if slope != 1:
            image = slope * image.astype(np.float64)
            image = image.astype(np.int16)
        image += np.int16(intercept)

        # Scope to center skull
        arr = np.array(iu.rescale2d(image) * 255, np.uint8)
        arr = iu.apply_clahe(arr)

        thresholds = [100, 50, 20]
        for th in thresholds:
            seg = arr.copy()
            seg[seg > th] = 255
            seg[seg <= th] = 0

            largest = np.array(iu.largest_cc(seg), dtype=np.uint8) * 255
            x, y, w, h = cv2.boundingRect(largest)
            img_arr = arr[y:y + h, x:x + w].copy()

            if th == thresholds[-1] or np.product(img_arr.shape) > 200 * 200:
                img = Image.fromarray(img_arr)
                _lbl = '_'.join([str(int(l)) for l in label])
                _name = _lbl + '-' + file.split('.')[0] + '.png'
                img.resize(resize_shape, Image.BILINEAR) \
                    .save(out_dir + os.sep + _name)
                break
    except Exception as e:
        print(file)
        traceback.print_exc()


import multiprocessing as mp
from functools import partial


def execute(mapping_file, images_dir, out_dir, resize_shape=None, limit=10e10, num_workers=1):
    print(mapping_file, images_dir)
    indices = load_indices(mapping_file=mapping_file, limit=limit)
    params = []
    mapper = partial(_save, images_dir=images_dir, out_dir=out_dir,
                     resize_shape=resize_shape)
    for ix, (file, label) in enumerate(indices, 1):
        params.append([ix, file, label])
    pool = mp.Pool(num_workers)
    pool.starmap(mapper, params)
    pool.close()
    pool.join()
