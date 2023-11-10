from pathlib import Path
import shutil

import nibabel
import numpy as np
import open3d
reg = open3d.pipelines.registration
from scipy import ndimage

import os, sys
sys.path.append(str(Path(os.getcwd()) / 'jawfrac_git'))
os.chdir(sys.path[-1])

from jawfrac_git.infer_mandibles import infer


def infer_mandibles(in_dir: Path):
    cbct_files = sorted(in_dir.glob('*.nii.gz'))
    images_nii = [nibabel.load(f) for f in cbct_files]

    preds = infer(
        Path(os.getcwd()).parent / 'checkpoints/mandibles.ckpt',        
        in_dir,
        interpolation='fast',
    )

    out = [
        {
            'cbct': np.asarray(img_nii.dataobj),
            'affine': img_nii.affine,
            'mask': pred[1],
        }
        for img_nii, pred in zip(images_nii, preds)
    ]

    for mandible in out:
        mask = mandible['mask']
        labels, max_label = ndimage.label(mask, ndimage.generate_binary_structure(3, 1))
        counts = ndimage.sum_labels(np.ones_like(labels), labels, range(1, max_label + 1))
        mask = labels == (np.argsort(counts)[-1] + 1)
        mandible['mask'] = mask

    return out


def crop_condyles(out_dir, mandibles: dict):
    condyle_slices = []
    for i, mandible in enumerate(mandibles):
        cbct, mask, affine = [mandible[k] for k in ['cbct', 'mask', 'affine']]
        orientation = nibabel.io_orientation(affine)
        axes = orientation[:, 0].astype(int)

        voxels = np.column_stack(np.nonzero(mask))
        bbox = np.concatenate((
            np.min(voxels, axis=0),
            np.max(voxels, axis=0),
        ))
        dims = bbox[3:] - bbox[:3]

        for right in [False, True]:
            split_bbox = np.array([
                [
                    right * (mask.shape[axes[0]] - 1) // 2,
                    0,
                    bbox[axes[2]] + dims[axes[2]] * 3 // 4,
                ],
                [
                    (1 + right) * (mask.shape[axes[0]] - 1) // 2,
                    mask.shape[axes[1]] - 1,
                    mask.shape[axes[2]] - 1,
                ],
            ])
            split_bbox = split_bbox[:, axes]
            slices = tuple([slice(l, r) for l, r in zip(split_bbox[0], split_bbox[1])])

            condyle = mask[slices]
            centroid = np.column_stack(np.nonzero(condyle)).mean(axis=0).astype(int)
            condyle_bbox = tuple([
                slice(max(coord - 100, 0), coord + 100)
                for coord in (centroid + split_bbox[0])
            ])

            image_crop = cbct[condyle_bbox]
            img = nibabel.Nifti1Image(image_crop.astype(np.int16), affine)

            name = f'{i}_{"right" if right else "left"}'
            nibabel.save(img, out_dir / f'{name}.nii.gz')
            condyle_slices.append(condyle_bbox)

    return condyle_slices


def infer_condyles(mandibles):
    out_dir = Path(os.getcwd()).parent
    bbox_slices = crop_condyles(out_dir, mandibles)

    preds = infer(        
        Path(os.getcwd()).parent / 'checkpoints/condyles_full.ckpt',     
        out_dir,
        channels_list=[32, 64, 128, 256],
        interpolation='slow',
    )

    out = [
        {
            'cbct': mandibles[i // 2]['cbct'],
            'mask': pred[1],
            'affine': mandibles[i // 2]['affine'],
        }
        for i, pred in enumerate(preds)
    ]
    for i, wol in enumerate(out):
        mask = wol['mask']
        mask = ndimage.binary_opening(mask, ndimage.generate_binary_structure(3, 3))
        labels, max_label = ndimage.label(mask, ndimage.generate_binary_structure(3, 1))
        counts = ndimage.sum_labels(np.ones_like(labels), labels, range(1, max_label + 1))
        mask = labels == (np.argsort(counts)[-1] + 1)

        wol['mask'] = np.zeros_like(wol['cbct'], dtype=bool)
        wol['mask'][bbox_slices[i]] = mask
    
    return out


def crop_mask_from_notch(condyle):
    orientation = nibabel.io_orientation(condyle['affine'])
    axes = orientation[:, 0].astype(int)

    condyle_2d = condyle['mask'].max(axis=axes[0])
    condyle_2d = ndimage.binary_closing(
        condyle_2d, ndimage.generate_binary_structure(2, 2),
        iterations=3,
    )

    if axes[1] < axes[2]:
        diffs = condyle_2d[:-1] ^ condyle_2d[1:]
        counts = diffs.sum(axis=0)
        notch_idx_x = counts.shape[0] - 1 - counts[::-1].argmax()
        while counts[notch_idx_x] > 2:
            if counts[notch_idx_x - 1] <= 2:
                break
            notch_idx_x -= 1
        notch_idx_y = np.nonzero(diffs[:, notch_idx_x])[0][1]
    else:
        diffs = condyle_2d[:, :-1] ^ condyle_2d[:, 1:]
        counts = diffs.sum(axis=1)
        notch_idx_x = counts.shape[0] - 1 - counts[::-1].argmax()
        while counts[notch_idx_x] > 2:
            if counts[notch_idx_x - 1] <= 2:
                break
            notch_idx_x -= 1
        notch_idx_y = np.nonzero(diffs[notch_idx_x])[0][1]

    mask = np.zeros_like(condyle['mask'], dtype=bool)
    slices = [slice(None) for _ in range(3)]
    slices[axes[1]] = slice(0, notch_idx_y)
    slices[axes[2]] = slice(notch_idx_x, mask.shape[axes[2]])
    mask[tuple(slices)] = True

    return mask


def main(in_dir: Path, out_dir: Path):
    mandibles = infer_mandibles(in_dir)
    condyles = infer_condyles(mandibles)

    condyles = [
        condyle['mask'] * crop_mask_from_notch(condyle)
        for condyle in condyles
    ]

    files = sorted(in_dir.glob('*.nii.gz'))
    for i, (file, mandible, left, right) in enumerate(
        zip(files, mandibles, condyles[::2], condyles[1::2]),
    ):
        mask = left | right

        img = nibabel.Nifti1Image(mask.astype(np.uint8), mandible['affine'])
        nibabel.save(img, out_dir / file.name)


if __name__ == '__main__':
    # in_dir = Path('/home/mkaailab/Documents/condyles/test')
    # out_dir = in_dir / 'out'


    in_dir = Path('/input')
    out_dir = Path('/output')
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(exist_ok=True)

    try:
        test_file = out_dir / 'test.txt'
        with open(test_file, 'w') as f:
            f.write('test')
            f.flush()
        test_file.unlink()
    except Exception:
        print('Give permission 777 to output folder.')
        exit()

    main(in_dir, out_dir)
