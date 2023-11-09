from pathlib import Path

import matplotlib.pyplot as plt
import nibabel
import numpy as np


root = Path('nnUNet_raw/Dataset005_CondyleNotchCoronoidCrops')

for seg_file in root.glob('labelsTr/*.nii.gz'):
    seg_nii = nibabel.load(seg_file)
    seg = np.asarray(seg_nii.dataobj)

    seg_sagittal = seg.max(axis=0)
    pos_diff = (seg_sagittal[:-1] - seg_sagittal[1:]) > 1
    neg_diff = (seg_sagittal[1:] - seg_sagittal[:-1]) > 1

    seg_deriv = pos_diff + neg_diff
    crosses = seg_deriv.sum(axis=0)

    notch_idx = (crosses > 2).argmax()


    k = 3

