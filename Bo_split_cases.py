from pathlib import Path

import nibabel
import numpy as np

root = Path('/mnt/diag/condyles/Cases')
out_dir = Path('nnUNet_raw/Dataset001_Condyles')

i = 0
for case_dir in sorted(root.glob('*')):
    for seg_file in case_dir.glob('*condyle.nii.gz'):
        seg_nii = nibabel.load(seg_file)
        seg = np.asarray(seg_nii.dataobj)

        image_file = '_'.join(seg_file.stem.split('_')[:-1] + ['image']) + '.nii.gz'
        image_nii = nibabel.load(seg_file.parent / image_file)
        image = np.asarray(image_nii.dataobj)

        for split_left in [True, False]:
            sagittal_slice = slice(
                split_left * image.shape[0] // 2,
                (1 + split_left) * image.shape[0] // 2,
            )

            seg_split = seg[sagittal_slice]
            seg_nii = nibabel.Nifti1Image(seg_split, seg_nii.affine)
            nibabel.save(seg_nii, out_dir / f'labelsTr/condyle_{i:03}.nii.gz')


            image_split = image[sagittal_slice]
            image_nii = nibabel.Nifti1Image(image_split, image_nii.affine)
            nibabel.save(image_nii, out_dir / f'imagesTr/condyle_{i:03}_0000.nii.gz')

            i += 1
