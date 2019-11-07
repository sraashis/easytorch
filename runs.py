import os

sep = os.sep
# --------------------------------------------------------------------------------------------
DRIVE = {
    'images_dir': 'data' + sep + 'DRIVE' + sep + 'images',
    'masks_dir': 'data' + sep + 'DRIVE' + sep + 'mask',
    'labels_dir': 'data' + sep + 'DRIVE' + sep + 'manual',
    'splits_dir': 'data' + sep + 'DRIVE' + sep + 'splits',
    'label_getter': lambda file_name: file_name.split('_')[0] + '_manual1.gif',
    'mask_getter': lambda file_name: file_name.split('_')[0] + '_mask.gif',
}

# --------------------------------------------------------------------------------------------
WIDE = {
    'images_dir': 'data' + sep + 'AV-WIDE' + sep + 'images',
    'labels_dir': 'data' + sep + 'AV-WIDE' + sep + 'manual',
    'label_getter': lambda file_name: file_name.split('.')[0] + '_vessels.png',
}

# ---------------------------------------------------------------------------------------------
STARE = {
    'images_dir': 'data' + sep + 'STARE' + sep + 'stare-images',
    'labels_dir': 'data' + sep + 'STARE' + sep + 'labels-ah',
    'label_getter': lambda file_name: file_name.split('.')[0] + '.ah.pgm',
}

# ------------------------------------------------------------------------------------------------
CHASEDB = {
    'images_dir': 'data' + sep + 'CHASEDB' + sep + 'images',
    'labels_dir': 'data' + sep + 'CHASEDB' + sep + 'manual',
    'label_getter': lambda file_name: file_name.split('.')[0] + '_1stHO.png',
}

# -------------------------------------------------------------------------------------------------
VEVIO_MOSAICS = {
    'images_dir': 'data' + sep + 'VEVIO' + sep + 'mosaics',
    'masks_dir': 'data' + sep + 'VEVIO' + sep + 'mosaics_masks',
    'labels_dir': 'data' + sep + 'VEVIO' + sep + 'mosaics_manual_01_bw',
    'label_getter': lambda file_name: 'bw_' + file_name.split('.')[0] + '_black.' + file_name.split('.')[1],
    'mask_getter': lambda file_name: 'mask_' + file_name
}

# ---------------------------------------------------------------------------------------------------------
VEVIO_FRAMES = {
    'images_dir': 'data' + sep + 'VEVIO' + sep + 'frames',
    'mask': 'data' + sep + 'VEVIO' + sep + 'frames_masks',
    'labels_dir': 'data' + sep + 'VEVIO' + sep + 'frames_manual_01_bw',
    'label_getter': lambda file_name: 'bw_' + file_name.split('.')[0] + '_black.' + file_name.split('.')[1],
    'mask_getter': lambda file_name: 'mask_' + file_name
}
# -------------------------------------------------------------------------------------------------------------
