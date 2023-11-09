from pathlib import Path
import shutil

import nibabel
import numpy as np
import pandas as pd
from tqdm import tqdm


if __name__ == '__main__':
    root = Path('/mnt/diag/Fabian')

    csv_path = root / 'Fabian overview teeth.csv'
    df = pd.read_csv(csv_path)
    for i, row in tqdm(list(df.iterrows())):
        # scan was checked
        if (
            isinstance(row['Complete mandible'], float) and
            isinstance(row['Complete teeth'], float)
        ):
            continue

        # no corruption
        if (
            not isinstance(row['Note'], float) and
            row['Note'] != 'bad orientation'
        ):
            continue

        # copy scan
        shutil.copy(
            root / row['Pseudonym'] / 'image.nii.gz',
            f'nnUNet_raw/Dataset002_Mandibles/imagesTr/mandibles_{i:03}_0000.nii.gz',
        )

        seg_nii = nibabel.load(root / row['Pseudonym'] / 'seg.nii.gz')
        seg = np.asarray(seg_nii.dataobj)
        seg = (seg == 2).astype(np.uint8)

        seg_nii = nibabel.Nifti1Image(seg, seg_nii.affine)
        nibabel.save(seg_nii, f'nnUNet_raw/Dataset002_Mandibles/labelsTr/mandibles_{i:03}.nii.gz')




