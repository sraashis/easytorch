import os

sep = os.sep
# --------------------------------------------------------------------------------------------
"""
DRIVE Dataset Reference:
Staal, J., Abramoff, M., Niemeijer, M., Viergever, M., and van Ginneken, B. (2004). 
Ridge based vessel segmentation in color images of the retina.
IEEE Transactions on Medical Imaging23, 501–509
"""
DRIVE = {
    'data_dir': 'DRIVE' + sep + 'images',
    'mask_dir': 'DRIVE' + sep + 'mask',
    'label_dir': 'DRIVE' + sep + 'OD_Segmentation',
    'split_dir': 'DRIVE' + sep + 'splits',
    'label_getter': lambda file_name: file_name.split('.')[0] + '_gt.tif',
    'mask_getter': lambda file_name: file_name.split('_')[0] + '_mask.gif',
}
"""
Estrada,  R.,  Tomasi,  C.,  Schmidler,  S. C.,  and Farsiu,  S. (2015).  
Tree topology estimation. IEEE Transactions on Pattern Analysis and Machine Intelligence
37, 1688–1701. doi:10.1109/TPAMI.2014.2592382116
"""
AV_WIDE = {
    'data_dir': 'AV-WIDE' + sep + 'images',
    'label_dir': 'AV-WIDE' + sep + 'OD_Segmentation',
    'split_dir': 'AV-WIDE' + sep + 'splits',
    'label_getter': lambda file_name: file_name.split('.')[0] + '_gt.png'
}