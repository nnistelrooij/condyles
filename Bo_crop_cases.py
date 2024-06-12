from pathlib import Path

import nibabel
import numpy as np
import pandas as pd
from scipy import ndimage
from tqdm import tqdm


if __name__ == '__main__':
    root = Path('nnUNet_raw/Dataset001_BoCondyles')
    out_dir = Path('nnUNet_raw/Dataset006_CondyleNotchCrops')

    (out_dir / 'labelsTr').mkdir(parents=True, exist_ok=True)
    (out_dir / 'imagesTr').mkdir(parents=True, exist_ok=True)

    df = pd.read_csv('/mnt/diag/CBCT/condyle_segmentation/Bo/dimensions segmentations.csv', dtype=object)

    notch_voxels = -1  # -1 for no notch crop

    
    for image_file, (i, row) in zip(
        tqdm(sorted(root.glob('**/*0000.nii.gz'))),
        df.iterrows(),
    ):
        if isinstance(row['Missing'], str) and (
            'Condyle' in row['Missing']
            or 'Notch' in row['Missing']
            or 'Coronoid' in row['Missing']
        ):
            continue

        condyle_file = image_file.parent.parent / 'labelsTr' / (image_file.stem[:-9] + '.nii.gz')

        image_nii = nibabel.load(image_file)
        image = np.asarray(image_nii.dataobj)
        
        condyle_nii = nibabel.load(condyle_file)
        condyle = np.asarray(condyle_nii.dataobj)

        labels, max_label = ndimage.label(condyle, ndimage.generate_binary_structure(3, 3))
        counts = ndimage.sum_labels(np.ones_like(labels), labels, range(1, max_label + 1))
        condyle = labels == (counts.argmax() + 1)
        coords = np.column_stack(np.nonzero(condyle))
        centroid = coords.mean(axis=0).astype(int)

        bbox = np.concatenate((
            np.maximum(0, centroid - 125),
            np.minimum(centroid + 125, condyle.shape),
        ))

        if notch_voxels == -1:
            bbox[2] = max(bbox[2], coords[:, 2].min())
        else:
            condyle_sagittal = condyle.max(axis=0)
            condyle_diffs = condyle_sagittal[:-1] ^ condyle_sagittal[1:]
            crosses = condyle_diffs.sum(axis=0)

            notch_idx = (crosses > 2).argmax()
            assert notch_idx >= notch_voxels
            bbox[2] = notch_idx - notch_voxels

        if isinstance(row['Missing'], str) and 'Coronoid' in row['Missing']:
            bbox[4] = min(bbox[4], coords[:, 1].max() + 1)

        crop_slices = ()
        for coord1, coord2 in zip(bbox[:3], bbox[3:]):
            crop_slices += (slice(coord1, coord2),)

        condyle_crop = condyle[crop_slices]
        condyle_nii = nibabel.Nifti1Image(condyle_crop.astype(np.uint8), condyle_nii.affine)
        nibabel.save(condyle_nii, out_dir / f'labelsTr/{condyle_file.name}')


        image_crop = image[crop_slices]
        image_nii = nibabel.Nifti1Image(image_crop, image_nii.affine)
        nibabel.save(image_nii, out_dir / f'imagesTr/{image_file.name}')
