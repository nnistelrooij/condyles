from pathlib import Path

import nibabel
import numpy as np
import pandas as pd
from tqdm import tqdm



def read_dat_file(dat_file: Path):
    with open(dat_file, 'r') as f:
        lines = f.readlines()

    shape = list(map(int, lines[0].split()))[::-1]
    spacing = list(map(float, lines[2].split()))[::-1]

    return shape, spacing


def read_volume(volume_dir: Path, key):
    dat_file = next(volume_dir.glob(f'{key}*.dat'))
    shape, spacing = read_dat_file(dat_file)

    raw_file = sorted(dat_file.parent.glob(dat_file.stem + '*'))[-1]
    if 'condyle' in raw_file.stem:
        raw_data = np.fromfile(raw_file, dtype=np.uint8)
        raw_data = (raw_data > 0).astype(np.uint8)
    elif 'grayscale' in raw_file.stem:
        raw_data = np.fromfile(raw_file, dtype=np.int16)
    else:
        raise ValueError()

    raw_data = raw_data.reshape(*shape)
    raw_data = np.transpose(raw_data, (2, 1, 0))[::-1, ::-1]

    affine = np.eye(4)
    affine[np.diag_indices(3)] = spacing[::-1]
    affine = np.round(affine, 3)

    return raw_data, affine


def process_volume(volume_dir: Path, moment: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    condyle_file = out_dir / f'{moment}_condyle.nii.gz'
    if condyle_file.exists():
        condyle_nii = nibabel.load(condyle_file)
        condyle = np.asarray(condyle_nii.dataobj)

        condyle_volume, _ = read_volume(volume_dir, 'condyle')
        condyle = condyle | condyle_volume

        condyle_nii = nibabel.Nifti1Image(condyle, condyle_nii.affine)
        nibabel.save(condyle_nii, condyle_file)

        return

    image_volume, image_affine = read_volume(volume_dir, 'grayscale')
    img = nibabel.Nifti1Image(image_volume, image_affine)
    nibabel.save(img, out_dir / f'{moment}_image.nii.gz')

    condyle_volume, condyle_affine = read_volume(volume_dir, 'condyle')
    img = nibabel.Nifti1Image(condyle_volume, condyle_affine)
    nibabel.save(img, out_dir / f'{moment}_condyle.nii.gz')


if __name__ == '__main__':
    root = Path('/mnt/diag/condyles')

    df = pd.read_excel(root / 'dimensions segmentations.xlsx')

    patients = df['Patient'][~pd.isna(df['Volume'])]
    _, index = np.unique(patients, return_index=True)
    patients = patients[np.sort(index)]
    patient2case = {patient: i + 1 for i, patient in enumerate(patients)}

    in_dir = root / 'Data'
    out_dir = root / 'Cases'

    t = tqdm(list(df.iterrows()))
    for i, row in t:
        volume_dir = in_dir / row['Volume']
        t.set_description(str(volume_dir))
        
        if not volume_dir.exists():
            continue

        process_volume(
            volume_dir,
            moment=row['Moment'],
            out_dir=out_dir / f'Case {patient2case[row["Patient"]]}',
        )
