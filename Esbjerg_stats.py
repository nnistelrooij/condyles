from pathlib import Path

import nibabel
import numpy as np


root = Path('/mnt/diag/condyles/Esbjerg/Cases')
root = Path('nnUNet_raw/Dataset003_CondyleCrops/labelsTr')

spacings, shapes = [], []
for nii_file in root.glob('*.nii.gz'):
    img = nibabel.load(nii_file)
    spacing = np.linalg.norm(img.affine[:, :3], axis=0)
    shape = img.header.get_data_shape()

    spacings.append(spacing)
    shapes.append(shape)

spacings = np.array(spacings)
print(spacings.min(), spacings.max())
print(spacings.min(axis=1) - spacings.max(axis=1))

shapes = np.array(shapes)
print(shapes.min(0))
