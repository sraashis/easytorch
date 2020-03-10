import os

sep = os.sep
# --------------------------------------------------------------------------------------------
DRIVE = {
    'images_dir': 'DRIVE' + sep + 'images',
    'masks_dir': 'DRIVE' + sep + 'mask',
    'labels_dir': 'DRIVE' + sep + 'manual',
    'splits_dir': 'DRIVE' + sep + 'splits',
    'label_getter': lambda file_name: file_name.split('_')[0] + '_manual1.gif',
    'mask_getter': lambda file_name: file_name.split('_')[0] + '_mask.gif',
}