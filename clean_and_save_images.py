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
    print(mapping_file, '...')
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


def run(mapping_file, images_dir, out_dir, resize_shape=None):
    indices = load_indices(mapping_file=mapping_file)
    for icount, (image_file, label) in enumerate(indices, 1):
        print('Saving {}/{}'.format(icount, len(indices)), end='\r')
        try:
            dcm = pydicom.dcmread(images_dir + os.sep + image_file)
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
            img_arr = np.array(iu.rescale2d(image) * 255, np.uint8)
            img_arr = iu.apply_clahe(img_arr)

            seg = img_arr.copy()
            seg[seg > 99] = 255
            seg[seg <= 99] = 0

            largest_cc = np.array(iu.largest_cc(seg), dtype=np.uint8) * 255
            x, y, w, h = cv2.boundingRect(largest_cc)
            img_arr = img_arr[y:y + h, x:x + w]

            img = Image.fromarray(img_arr)
            _name = '_'.join() + '-' + image_file.split['.'][0] + '.png'
            img.resize(resize_shape, Image.BILINEAR) \
                .save(out_dir + os.sep + _name)
            print('H')
        except Exception as e:
            print('###' + image_file, str(e))
