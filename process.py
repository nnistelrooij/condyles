from pathlib import Path
import shutil
from typing import Literal

import nibabel
import numpy as np
import open3d
reg = open3d.pipelines.registration
from scipy import ndimage

import os, sys
sys.path.append(str(Path(os.getcwd()) / 'jawfrac_git'))
os.chdir(sys.path[-1])

from jawfrac_git.infer_mandibles import infer
from register import register


def infer_mandibles(in_dir):
    preds = infer(
        Path(os.getcwd()).parent / 'checkpoints/mandibles.ckpt',        
        in_dir,
        interpolation='none',
    )
    out = {
        'baseline': {'cbct': preds[0][0], 'mask': preds[0][1]},
        'recall': {'cbct': preds[1][0], 'mask': preds[1][1]},
    }

    for moment in out:
        mask = out[moment]['mask']
        labels, max_label = ndimage.label(mask, ndimage.generate_binary_structure(3, 1))
        counts = ndimage.sum_labels(np.ones_like(labels), labels, range(1, max_label + 1))
        mask = labels == (np.argsort(counts)[-1] + 1)
        out[moment]['mask'] = mask

    return out


def crop_condyles(out_dir, mandibles: dict):
    condyle_slices = {}
    for moment in mandibles:
        cbct, mask = [mandibles[moment][k] for k in ['cbct', 'mask']]
        voxels = np.column_stack(np.nonzero(mask))
        bbox = np.concatenate((
            np.min(voxels, axis=0),
            np.max(voxels, axis=0),
        ))
        dims = bbox[3:] - bbox[:3]

        for right in [False, True]:
            split_bbox = [
                right * (mask.shape[1] - 1) // 2,
                0,
                bbox[2] + dims[2] * 3 // 4,
                (1 + right) * (mask.shape[1] - 1) // 2,
                mask.shape[0] - 1,
                mask.shape[2] - 1,
            ]
            slices = tuple([slice(l, r) for l, r in zip(split_bbox[:3], split_bbox[3:])])

            condyle = mask[slices]
            centroid = np.column_stack(np.nonzero(condyle)).mean(axis=0).astype(int)
            condyle_bbox = tuple([
                slice(max(coord - 100, 0), coord + 100)
                for coord in (centroid + split_bbox[:3])
            ])

            image_crop = cbct[condyle_bbox]
            affine = np.eye(4)
            affine[np.diag_indices(3)] = 0.4
            img = nibabel.Nifti1Image(image_crop.astype(np.int16), affine)

            name = f'{moment}_{"right" if right else "left"}'
            nibabel.save(img, out_dir / f'{name}.nii.gz')
            condyle_slices[name] = condyle_bbox

    return condyle_slices


def infer_condyles(mandibles):
    out_dir = Path(os.getcwd()).parent
    bbox_slices = crop_condyles(out_dir, mandibles)

    preds = infer(        
        Path(os.getcwd()).parent / 'checkpoints/condyles_full.ckpt',     
        out_dir,
        channels_list=[32, 64, 128, 256],
        interpolation='none',
    )

    out = {
        key: {'cbct': mandibles[key.split('_')[0]]['cbct'], 'mask': preds[i][1]}
        for i, key in enumerate(bbox_slices)
    }
    for moment in out:
        mask = out[moment]['mask']
        mask = ndimage.binary_opening(mask, ndimage.generate_binary_structure(3, 3))
        labels, max_label = ndimage.label(mask, ndimage.generate_binary_structure(3, 1))
        counts = ndimage.sum_labels(np.ones_like(labels), labels, range(1, max_label + 1))
        mask = labels == (np.argsort(counts)[-1] + 1)

        out[moment]['mask'] = np.zeros_like(out[moment]['cbct'], dtype=bool)
        out[moment]['mask'][bbox_slices[moment]] = mask
    
    return out


def transform(baseline, recall, affine, method: Literal['slow', 'fast']='fast'):
    if method == 'slow':
        out = ndimage.affine_transform(
            input=baseline.astype(np.float32),
            matrix=np.linalg.inv(affine),
            output_shape=recall.shape,
        )
        return out >= 0.5

    # determine crop of baseline around condyle
    slices = ndimage.find_objects(baseline)[0]
    dims = [slc.stop - slc.start for slc in slices]
    slices_crop = ()
    for slc, crop_dim, dim in zip(slices, dims, baseline.shape):
        slices_crop += (slice(
            max(slc.start - crop_dim // 2, 0),
            min(dim, slc.stop + crop_dim // 2),
        ),)
    baseline_coords = np.array([
        [slc.start for slc in slices_crop],
        [slc.stop for slc in slices_crop],
    ])
    baseline_crop = baseline[slices_crop]

    # determine translate - affine - inv_translate transformation
    affine_translate = np.eye(4)
    affine_translate[:3, 3] = baseline_coords[0]
    affine_crop = np.linalg.inv(affine_translate) @ affine @ affine_translate

    # determine crop of recall around transformed condyle
    recall_coords = np.array([
        [slc.start for slc in slices_crop] + [1],
        [slc.stop for slc in slices_crop] + [1],
    ])
    recall_coords = (recall_coords @ affine.T)[:, :-1].astype(int)
    recall_coords[0] = np.maximum(recall_coords[0], 0)
    recall_coords[1] = np.minimum(np.array(recall.shape), recall_coords[1])
    recall_crop = tuple([
        slice(start, stop) for start, stop in recall_coords.T
    ])
    recall_dims = recall_coords[1] - recall_coords[0]

    # transform baseline condyle and insert as recall crop
    baseline_trans = ndimage.affine_transform(
        input=baseline_crop.astype(np.float32),
        matrix=np.linalg.inv(affine_crop[:3, :3]),
        output_shape=recall_dims,
    )
    out = recall.copy().astype(float)
    out[recall_crop] = baseline_trans

    return out >= 0.5


def crop_mask_from_notch(condyle):
    condyle_2d = condyle.max(axis=0)
    condyle_2d = ndimage.binary_closing(
        condyle_2d, ndimage.generate_binary_structure(2, 2),
        iterations=3,
    )

    diffs = condyle_2d[:-1] ^ condyle_2d[1:]
    counts = diffs.sum(axis=0)
    notch_idx_x = counts.shape[0] - 1 - counts[::-1].argmax()
    while counts[notch_idx_x] > 2:
        if counts[notch_idx_x - 1] <= 2:
            break
        notch_idx_x -= 1
    notch_idx_y = np.nonzero(diffs[:, notch_idx_x])[0][1]

    mask = np.zeros_like(condyle, dtype=bool)
    mask[:, :notch_idx_y, notch_idx_x:] = True

    return mask


def main(in_dir, out_dir):
    mandibles = infer_mandibles(in_dir)
    affine_mandible = register(
        mandibles['baseline']['mask'],
        mandibles['recall']['mask'],
        voxel_size=4,
    )

    condyles = infer_condyles(mandibles)
    baseline_left_trans = transform(
        condyles['baseline_left']['mask'],
        condyles['recall_left']['mask'],
        affine_mandible,
    )
    baseline_right_trans = transform(
        condyles['baseline_right']['mask'],
        condyles['recall_right']['mask'],
        affine_mandible,
    )

    left_mask = crop_mask_from_notch(condyles['recall_left']['mask'])
    right_mask = crop_mask_from_notch(condyles['recall_right']['mask'])
    affine_left = register(
        baseline_left_trans * ~left_mask,
        condyles['recall_left']['mask'] * ~left_mask,
        voxel_size=2,
    ) @ affine_mandible
    affine_right = register(
        baseline_right_trans * ~right_mask,
        condyles['recall_right']['mask'] * ~right_mask,
        voxel_size=2,
    ) @ affine_mandible

    left_mask = crop_mask_from_notch(condyles['recall_left']['mask'])
    right_mask = crop_mask_from_notch(condyles['recall_right']['mask'])
    condyles['baseline_left']['mask'] = transform(
        condyles['baseline_left']['mask'],
        condyles['recall_left']['mask'],
        affine_left,
    ) * left_mask
    condyles['recall_left']['mask'] *= left_mask
    condyles['baseline_right']['mask'] = transform(
        condyles['baseline_right']['mask'],
        condyles['recall_right']['mask'],
        affine_right,
    ) * right_mask
    condyles['recall_right']['mask'] *= right_mask

    for moment in condyles:
        num_voxels = condyles[moment]['mask'].sum()
        print(f'{moment} (mm3): {num_voxels * 0.4**3:.0f}')

    
    affine = np.eye(4)
    affine[np.diag_indices(3)] = 0.4

    img = nibabel.Nifti1Image(mandibles['recall']['cbct'].astype(np.int16), affine)
    nibabel.save(img, out_dir / f'cbct.nii.gz')

    baseline = condyles['baseline_left']['mask'] | condyles['baseline_right']['mask']
    img = nibabel.Nifti1Image(baseline.astype(np.uint8), affine)
    nibabel.save(img, out_dir / 'baseline.nii.gz')

    recall = condyles['recall_left']['mask'] | condyles['recall_right']['mask']
    img = nibabel.Nifti1Image(recall.astype(np.uint8), affine)
    nibabel.save(img, out_dir / 'recall.nii.gz')



if __name__ == '__main__':
    # in_dir = Path('/home/mkaailab/Documents/condyles/test')
    # out_dir = in_dir / 'out'

    in_dir = Path('/input')
    in_files = sorted(in_dir.glob('*.nii.gz'))
    assert len(in_files) == 2, 'Please provide exactly two nii.gz files.'
    assert 'baseline' in in_files[0].stem, 'Please use "baseline" in baseline scan filename.'
    assert 'recall' in in_files[1].stem, 'Please use "recall" in recall scan filename.'

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
