from pathlib import Path

import nibabel
import numpy as np


root = Path('/mnt/diag/condyles/Esbjerg')

in_dir = root / 'Cases'
out_dir = root / 'Crops'

crop_size_mm = 100
out_dir.mkdir(parents=True, exist_ok=True)

i = 0
for image_file in sorted(in_dir.glob('*0000.nii.gz')):
    image_nii = nibabel.load(image_file)
    image = np.asarray(image_nii.dataobj)
    spacing = np.linalg.norm(image_nii.affine[:, :3], axis=0)

    seg_file = in_dir / f'{image_file.stem.split("_")[0]}.nii.gz'
    if not seg_file.exists():
        continue

    seg_nii = nibabel.load(seg_file)
    seg = np.asarray(seg_nii.dataobj)

    for split_left in [True, False]:
        sagittal_slice = slice(
            split_left * image.shape[0] // 2,
            (1 + split_left) * image.shape[0] // 2,
        )

        image_split = image[:, sagittal_slice]
        seg_split = seg[:, sagittal_slice]

        coords = np.column_stack(np.nonzero(seg_split))
        centroid = coords.mean(axis=0).astype(int)

        crop_size_voxels = (crop_size_mm / spacing / 2).astype(int)
        slices = ()
        for j in range(3):
            slices += (slice(
                max(0, centroid[j] - crop_size_voxels[j]),
                min(centroid[j] + crop_size_voxels[j], image_split.shape[j])
            ),)

        image_crop = image_split[slices]
        img = nibabel.Nifti1Image(image_crop, image_nii.affine)
        nibabel.save(img, out_dir / f'{seg_file.stem}_{i}_0000.nii.gz')
        
        i += 1
