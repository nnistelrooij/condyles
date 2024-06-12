import multiprocessing as mp
from pathlib import Path

import nibabel
import numpy as np
from tqdm import tqdm


def process_case(case_dir: Path):
    for i, seg_file in enumerate(sorted(case_dir.glob('*condyle.nii.gz'))):
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
            nibabel.save(seg_nii, out_dir / f'labelsTr/condyle_{int(case_dir.name.split(" ")[-1]):02}{2*i + (not split_left)}9.nii.gz')


            image_split = image[sagittal_slice]
            image_nii = nibabel.Nifti1Image(image_split, image_nii.affine)
            nibabel.save(image_nii, out_dir / f'imagesTr/condyle_{int(case_dir.name.split(" ")[-1]):02}{2*i + (not split_left)}9_0000.nii.gz')

    return case_dir


if __name__ == '__main__':
    root = Path('/mnt/diag/CBCT/condyle_segmentation/Bo/Cases')

    out_dir = Path('nnUNet_raw/Dataset001_BoCondyles')
    (out_dir / 'labelsTr').mkdir(parents=True, exist_ok=True)
    (out_dir / 'imagesTr').mkdir(parents=True, exist_ok=True)

    case_dirs = sorted(root.glob('*'), key=lambda p: int(p.name.split(' ')[-1]))
    with mp.Pool(16) as p:
        t = tqdm(p.imap_unordered(process_case, case_dirs), total=len(case_dirs))
        for case_dir in t:
            t.set_description(case_dir.as_posix())
