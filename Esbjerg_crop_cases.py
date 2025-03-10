from functools import partial
import multiprocessing as mp
from pathlib import Path

import nibabel
import numpy as np
import pandas as pd
from scipy import ndimage
from tqdm import tqdm


def process_volumes(df: pd.DataFrame, cbct_file: Path):
    patient = cbct_file.parent.name.split('_')[0]
    moment = cbct_file.name.split('_')[0]

    image_nii = nibabel.load(cbct_file)
    image = np.asarray(image_nii.dataobj)
    spacing = np.linalg.norm(image_nii.affine[:, :3], axis=0)

    condyle_file = cbct_file.parent / f'{cbct_file.stem.split("_")[0]}_Condylar mask.nii.gz'
    mandible_file = cbct_file.parent / f'{cbct_file.stem.split("_")[0]}_Mandibular mask.nii.gz'
    if not condyle_file.exists():
        return cbct_file

    condyle_nii = nibabel.load(condyle_file)
    condyle = np.asarray(condyle_nii.dataobj)

    mandible_nii = nibabel.load(mandible_file)
    mandible = np.asarray(mandible_nii.dataobj)

    offset = 2 * (moment == 'Recall')
    for split_left in [True, False]:
        missed_df = df[
            (df['Patient'] == int(patient))
            & (df['Moment'] == moment)
            & (df['Side'] == ('Left' if split_left else 'Right'))
        ]
        if missed_df.shape[0] > 0 and not pd.isna(missed_df['Not fixed']).any():
            continue


        sagittal_slice = slice(
            split_left * image.shape[0] // 2,
            (1 + split_left) * image.shape[0] // 2,
        )

        image_split = image[:, sagittal_slice]
        condyle_split = condyle[:, sagittal_slice]
        mandbible_split = mandible[:, sagittal_slice]

        coords = np.column_stack(np.nonzero(condyle_split))
        centroid = coords.mean(axis=0).astype(int)

        crop_size_voxels = (crop_size_mm / spacing / 2).astype(int)
        slices = ()
        for j in range(3):
            amin = 0 if j < 2 else centroid[2] - int(down_mm / spacing[2])
            slices += (slice(
                max(amin, centroid[j] - crop_size_voxels[j]),
                min(centroid[j] + crop_size_voxels[j], image_split.shape[j])
            ),)

        image_crop = image_split[slices]
        img = nibabel.Nifti1Image(image_crop, image_nii.affine)
        nibabel.save(img, out_dir / f'imagesTr/condyle_{int(patient):02}{offset // 2}{offset % 2}_0000.nii.gz')
        
        mandible_crop = mandbible_split[slices]
        labels, max_label = ndimage.label(mandible_crop)
        try:
            counts = ndimage.sum_labels(np.ones_like(labels), labels, range(1, max_label + 1))
        except Exception:
            print(cbct_file)
        mandible_crop = labels == (counts.argmax() + 1)
        img = nibabel.Nifti1Image(mandible_crop.astype(np.uint16), image_nii.affine)
        nibabel.save(img, out_dir / f'labelsTr/condyle_{int(patient):02}{offset // 2}{offset % 2}.nii.gz')

        offset += 1

    return cbct_file


if __name__ == '__main__':
    root = Path('/mnt/diag/CBCT/condyle_segmentation/Esbjerg')

    df = pd.read_excel(root / 'missings.xlsx')

    out_dir = Path('nnUNet_raw/Dataset002_EsbjergCondyles')

    crop_size_mm = 100
    down_mm = 15
    (out_dir / 'imagesTr').mkdir(parents=True, exist_ok=True)
    (out_dir / 'labelsTr').mkdir(parents=True, exist_ok=True)

    cbct_files = sorted(root.glob('**/*CBCT.nii.gz'))
    with mp.Pool(16) as p:
        i = p.imap_unordered(partial(process_volumes, df), cbct_files)
        t = tqdm(i, total=len(cbct_files))
        for cbct_file in t:
            t.set_description(cbct_file.as_posix())
