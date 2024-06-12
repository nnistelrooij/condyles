import copy
from collections import defaultdict
from functools import partial
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import tempfile
import traceback

import nibabel
import numpy as np
import open3d
reg = open3d.pipelines.registration
from scipy import ndimage
import torch
from tqdm import tqdm

import os, sys
sys.path.append(str(Path(os.getcwd()) / 'jawfrac'))
os.chdir(sys.path[-1])

from jawfrac.infer_mandibles import infer
from register import (
    intensity_registration,
    surface_registration,
    transform,
    voxel_registration,
)


def save_volumes(out_dir: Path, volumes, suffix=''):
    affine = np.eye(4)
    affine[np.diag_indices(3)] = 0.4

    flat_volumes = defaultdict(list)
    for k, volume in volumes.items():
        flat_volumes[k.split('_')[0]].append(volume)

    for key, volumes in flat_volumes.items():
        out_file = out_dir / f'{key}_cbct.nii.gz'
        if not out_file.exists():
            img = nibabel.Nifti1Image(volumes[0]['cbct'].astype(np.int16), affine)
            nibabel.save(img, out_file)

        seg = np.zeros_like(volumes[0]['mask'], dtype=int)
        probs = np.zeros_like(volumes[0]['probs'], dtype=np.float32)
        for i, volume in enumerate(volumes, 1):
            seg += i * volume['mask']
            probs = np.maximum(probs, volume['probs'])

        img = nibabel.Nifti1Image(seg.astype(np.uint8), affine)
        nibabel.save(img, out_dir / f'{key}_seg{f"_{suffix}" if suffix else ""}.nii.gz')

        img = nibabel.Nifti1Image(probs, affine)
        nibabel.save(img, out_dir / f'{key}_probs{f"_{suffix}" if suffix else ""}.nii.gz')


def infer_mandibles(in_dir: Path, out_dir: Path):
    if (out_dir / 'baseline_seg_mandible.nii.gz').exists():
        baseline_nii = nibabel.load(out_dir / 'baseline_cbct.nii.gz')
        baseline_cbct = np.asarray(baseline_nii.dataobj)
        baseline_nii = nibabel.load(out_dir / 'baseline_probs_mandible.nii.gz')
        baseline_probs = np.asarray(baseline_nii.dataobj)
        baseline_nii = nibabel.load(out_dir / 'baseline_seg_mandible.nii.gz')
        baseline_mandible = np.asarray(baseline_nii.dataobj)

        recall_nii = nibabel.load(out_dir / 'recall_cbct.nii.gz')
        recall_cbct = np.asarray(recall_nii.dataobj)
        recall_nii = nibabel.load(out_dir / 'recall_probs_mandible.nii.gz')
        recall_probs = np.asarray(recall_nii.dataobj)
        recall_nii = nibabel.load(out_dir / 'recall_seg_mandible.nii.gz')
        recall_mandible = np.asarray(recall_nii.dataobj)

        out = {
            'baseline': {'cbct': baseline_cbct, 'probs': baseline_probs, 'mask': baseline_mandible},
            'recall': {'cbct': recall_cbct, 'probs': recall_probs, 'mask': recall_mandible},
        }
        return out


    preds = infer(
        checkpoint=Path(os.getcwd()).parent / 'checkpoints/mandibles.ckpt',        
        root=in_dir,
        interpolation='none',
    )
    out = {
        'baseline': {'cbct': preds[0][0], 'probs': preds[0][1].numpy()},
        'recall': {'cbct': preds[1][0], 'probs': preds[1][1].numpy()},
    }

    for moment in out:
        mask = out[moment]['probs'] >= 0.5
        mask = select_instance(mask)
        out[moment]['mask'] = mask

    return out


def crop_condyles(
    out_dir,
    mandibles: dict,
    crop_size: int=150,
):
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
                slice(max(coord - crop_size // 2, 0), coord + crop_size // 2)
                for coord in (centroid + split_bbox[:3] + [0, 10, 10])
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
    with tempfile.TemporaryDirectory() as f:
        out_dir = Path(f)
        bbox_slices = crop_condyles(out_dir, mandibles)

        preds = infer(
            checkpoint=Path(os.getcwd()).parent / 'checkpoints/condyles_all.ckpt',     
            root=out_dir,
            # channels_list=[32, 64, 128, 256],
            interpolation='none',
        )

    out = {
        key: {'cbct': mandibles[key.split('_')[0]]['cbct'], 'probs': preds[i][1]}
        for i, key in enumerate(bbox_slices)
    }
    for moment in out:
        probs = out[moment]['probs'].numpy()
        mask = select_instance(probs >= 0.5)

        out[moment]['mask'] = np.zeros_like(out[moment]['cbct'], dtype=bool)
        out[moment]['mask'][bbox_slices[moment]] = mask

        out[moment]['probs'] = np.zeros_like(out[moment]['cbct'], dtype=np.float32)
        out[moment]['probs'][bbox_slices[moment]] = probs
    
    return out


def notch_idxs(condyle):
    condyle_2d = condyle.max(axis=0)
    condyle_2d = ndimage.binary_closing(
        condyle_2d, ndimage.generate_binary_structure(2, 2),
        iterations=3,
    )
    condyle_2d = select_instance(condyle_2d)

    diffs = condyle_2d[:-1] ^ condyle_2d[1:]
    counts = diffs.sum(axis=0)
    notch_idx_x = counts.shape[0] - 1 - (counts[::-1] > 2).argmax()
    while counts[notch_idx_x] > 2:
        if counts[notch_idx_x - 1] <= 2:
            break
        notch_idx_x -= 1
    notch_idx_y = np.nonzero(diffs[:, notch_idx_x])[0][1]

    return notch_idx_y, notch_idx_x


def select_instance(mask):
    labels, max_label = ndimage.label(mask, ndimage.generate_binary_structure(mask.ndim, 1))
    counts = ndimage.sum_labels(np.ones_like(labels), labels, range(1, max_label + 1))
    out = labels == (np.argsort(counts)[-1] + 1)

    return out


def select_rami(mandibles, crop_size=(120, 120, 120)):
    out = {}
    for moment in mandibles:
        cbct, mask = [mandibles[moment][k] for k in ['cbct', 'mask']]

        for right in [False, True]:
            split_bbox = [
                right * (mask.shape[1] - 1) // 2,
                0,
                0,
                (1 + right) * (mask.shape[1] - 1) // 2,
                mask.shape[0] - 1,
                mask.shape[2] - 1,
            ]
            split_slices = tuple([slice(l, r) for l, r in zip(split_bbox[:3], split_bbox[3:])])

            notch_idx_y, notch_idx_x = notch_idxs(mask[split_slices])
            notch_idx_x += split_slices[2].start
            slices = split_slices[:-1] + (slice(0, notch_idx_x),)

            
            cutoff_y_idxs = (
                mask[slices].max(axis=0)[:, -1].argmax(),
                mask.shape[1] - 1 - mask[slices].max(axis=0)[::-1, -1].argmax(),
            )
            height = mask[split_slices].max(axis=0)[notch_idx_y].sum()
            cutoff = (
                int(cutoff_y_idxs[0] + (cutoff_y_idxs[1] - cutoff_y_idxs[0]) / 2),
                int(notch_idx_x - height / 2),
            )
            cutoff_slices = (
                slice(None), slice(cutoff[0], mask.shape[1]), slice(0, cutoff[1]),
            )



            condyle_slices = (
                slice(None), slice(0, notch_idx_y), slice(notch_idx_x, mask.shape[2]),
            )


            cutoff_mask1 = np.zeros_like(mask[split_slices])
            cutoff_mask1[condyle_slices] = True
            cutoff_mask1[cutoff_slices] = True
            cutoff_mask1[..., :int(notch_idx_x - height * 2 / 3)] = True

            bordered_rami_mask = mask.copy()[split_slices]
            bordered_rami_mask[condyle_slices] = True
            bordered_rami_mask[cutoff_slices] = True
            bordered_rami_mask[..., :int(notch_idx_x - height * 2 / 3)] = True

            filled_rami_mask1 = ndimage.binary_fill_holes(bordered_rami_mask, np.ones((3, 3, 3)))
            filled_rami_mask1[condyle_slices] = False
            filled_rami_mask1[cutoff_slices] = False
            filled_rami_mask1[..., :int(notch_idx_x - height * 2 / 3)] = False


            cutoff_mask2 = np.zeros_like(mask[split_slices])
            cutoff_mask2[cutoff_slices] = True
            cutoff_mask2[..., :int(notch_idx_x - height * 2 / 3)] = True

            bordered_rami_mask = mask.copy()[split_slices]
            bordered_rami_mask[cutoff_slices] = True
            bordered_rami_mask[..., :int(notch_idx_x - height * 2 / 3)] = True

            filled_rami_mask2 = ndimage.binary_fill_holes(bordered_rami_mask, np.ones((3, 3, 3)))
            filled_rami_mask2[cutoff_slices] = False
            filled_rami_mask2[..., :int(notch_idx_x - height * 2 / 3)] = False

            out_masks = np.zeros((4, *mask.shape))
            out_masks[0, split_slices[0], split_slices[1], split_slices[2]] = filled_rami_mask1
            out_masks[1, split_slices[0], split_slices[1], split_slices[2]] = cutoff_mask1
            out_masks[2, split_slices[0], split_slices[1], split_slices[2]] = filled_rami_mask2
            out_masks[3, split_slices[0], split_slices[1], split_slices[2]] = cutoff_mask2

            moment_dir = f'{moment}_{"right" if right else "left"}'
            out[moment_dir] = {
                'cbct': cbct,
                'mask': out_masks,
            }
            
    return out


def register_rami(mandibles, condyles, method='surface', verbose: bool=False):
    mandibles['baseline']['mask'] |= condyles['baseline_left']['mask']
    mandibles['baseline']['mask'] |= condyles['baseline_right']['mask']

    mandibles['recall']['mask'] |= condyles['recall_left']['mask']
    mandibles['recall']['mask'] |= condyles['recall_right']['mask']

    rami = select_rami(mandibles)

    if method == 'intensity':
        condyles['baseline_left']['mask'] = np.zeros_like(mandibles['recall']['mask'])
        condyles['baseline_left']['mask'][rami['recall_left']['slices']] = intensity_registration(
            rami['baseline_left'],
            rami['recall_left'],
            verbose=verbose,
        )

        condyles['baseline_right']['mask'] = np.zeros_like(mandibles['recall']['mask'])
        condyles['baseline_right']['mask'][rami['recall_right']['slices']] = intensity_registration(
            rami['baseline_right'],
            rami['recall_right'],
            verbose=verbose,
        )
    elif method == 'surface':
        affine_left = surface_registration(
            rami['baseline_left']['mask'],
            rami['recall_left']['mask'],
            mandibles['recall']['mask'],
            verbose=verbose,
        )

        affine_right = surface_registration(
            rami['baseline_right']['mask'],
            rami['recall_right']['mask'],
            mandibles['recall']['mask'],
            verbose=verbose,
        )

        condyles = transform_condyles(
            condyles, affine_left, affine_right,
        )
    else:
        raise ValueError(f'Unknown method: {method}')

    return condyles


def transform_condyles(
    condyles, affine_left, affine_right,
):
    condyles = copy.deepcopy(condyles)
    condyles['baseline_left']['mask'] = transform(
        condyles['baseline_left']['mask'],
        condyles['recall_left']['mask'],
        affine_left,
    ) >= 0.5
    condyles['baseline_right']['mask'] = transform(
        condyles['baseline_right']['mask'],
        condyles['recall_right']['mask'],
        affine_right,
    ) >= 0.5

    condyles['baseline_left']['probs'] = transform(
        condyles['baseline_left']['probs'],
        condyles['recall_left']['probs'],
        affine_left,
    )
    condyles['baseline_right']['probs'] = transform(
        condyles['baseline_right']['probs'],
        condyles['recall_right']['probs'],
        affine_right,
    )

    return condyles


def select_condyles(condyles):
    condyles = copy.deepcopy(condyles)

    
    notch_idx_y, notch_idx_x = notch_idxs(condyles['baseline_left']['mask'])
    condyles['baseline_left']['mask'][:, notch_idx_y:] = False
    condyles['baseline_left']['mask'][:, :, :notch_idx_x] = False
    condyles['baseline_left']['mask'] = ndimage.gaussian_filter(
        condyles['baseline_left']['mask'].astype(float), 1.2,
    ) >= 0.5
    condyles['baseline_left']['mask'] = select_instance(condyles['baseline_left']['mask'])
    condyles['baseline_left']['probs'][:, notch_idx_y:] = 0
    condyles['baseline_left']['probs'][:, :, :notch_idx_x] = 0

    condyles['recall_left']['mask'][:, notch_idx_y:] = False
    condyles['recall_left']['mask'][:, :, :notch_idx_x] = False
    condyles['recall_left']['mask'] = ndimage.gaussian_filter(
        condyles['recall_left']['mask'].astype(float), 1.2,
    ) >= 0.5
    condyles['recall_left']['mask'] = select_instance(condyles['recall_left']['mask'])
    condyles['recall_left']['probs'][:, notch_idx_y:] = 0
    condyles['recall_left']['probs'][:, :, :notch_idx_x] = 0


    notch_idx_y, notch_idx_x = notch_idxs(condyles['baseline_right']['mask'])
    condyles['baseline_right']['mask'][:, notch_idx_y:] = False
    condyles['baseline_right']['mask'][:, :, :notch_idx_x] = False
    condyles['baseline_right']['mask'] = ndimage.gaussian_filter(
        condyles['baseline_right']['mask'].astype(float), 1.2,
    ) >= 0.5
    condyles['baseline_right']['mask'] = select_instance(condyles['baseline_right']['mask'])
    condyles['baseline_right']['probs'][:, notch_idx_y:] = 0
    condyles['baseline_right']['probs'][:, :, :notch_idx_x] = 0

    condyles['recall_right']['mask'][:, notch_idx_y:] = False
    condyles['recall_right']['mask'][:, :, :notch_idx_x] = False
    condyles['recall_right']['mask'] = ndimage.gaussian_filter(
        condyles['recall_right']['mask'].astype(float), 1.2,
    ) >= 0.5
    condyles['recall_right']['mask'] = select_instance(condyles['recall_right']['mask'])
    condyles['recall_right']['probs'][:, notch_idx_y:] = 0
    condyles['recall_right']['probs'][:, :, :notch_idx_x] = 0

    return condyles


def interpolate_condyles(in_dir: Path, condyles):
    condyles = copy.deepcopy(condyles)

    recall_path = sorted(in_dir.glob('*.nii.gz'))[1]
    recall_nii = nibabel.load(recall_path)
    
    recall_orientation = nibabel.io_orientation(recall_nii.affine).astype(int)
    # condyles['baseline_left']['probs'] = nibabel.apply_orientation(
    #     condyles['baseline_left']['probs'],
    #     recall_orientation,
    # )
    # condyles['baseline_right']['probs'] = nibabel.apply_orientation(
    #     condyles['baseline_right']['probs'],
    #     recall_orientation,
    # )
    # condyles['recall_left']['probs'] = nibabel.apply_orientation(
    #     condyles['recall_left']['probs'],
    #     recall_orientation,
    # )
    # condyles['recall_right']['probs'] = nibabel.apply_orientation(
    #     condyles['recall_right']['probs'],
    #     recall_orientation,
    # )


    recall_shape = recall_nii.header.get_data_shape()
    recall_shape = np.array(recall_shape)[recall_orientation[:, 0]]
    recall_shape = tuple(recall_shape.tolist())
    condyles['baseline_left']['probs'] = torch.nn.functional.interpolate(
        input=torch.from_numpy(condyles['baseline_left']['probs'])[None, None],
        size=recall_shape,
        mode='trilinear',
    )[0, 0].numpy()
    condyles['baseline_right']['probs'] = torch.nn.functional.interpolate(
        input=torch.from_numpy(condyles['baseline_right']['probs'])[None, None],
        size=recall_shape,
        mode='trilinear',
    )[0, 0].numpy()
    condyles['recall_left']['probs'] = torch.nn.functional.interpolate(
        input=torch.from_numpy(condyles['recall_left']['probs'])[None, None],
        size=recall_shape,
        mode='trilinear',
    )[0, 0].numpy()
    condyles['recall_right']['probs'] = torch.nn.functional.interpolate(
        input=torch.from_numpy(condyles['recall_right']['probs'])[None, None],
        size=recall_shape,
        mode='trilinear',
    )[0, 0].numpy()
    
    
    condyles['baseline_left']['mask'] = ndimage.gaussian_filter(
        condyles['baseline_left']['probs'], 1.2,
    ) >= 0.5
    condyles['baseline_left']['mask'] = select_instance(condyles['baseline_left']['mask'])
    condyles['recall_left']['mask'] = ndimage.gaussian_filter(
        condyles['recall_left']['probs'], 1.2,
    ) >= 0.5
    condyles['recall_left']['mask'] = select_instance(condyles['recall_left']['mask'])
    condyles['baseline_right']['mask'] = ndimage.gaussian_filter(
        condyles['baseline_right']['probs'], 1.2,
    ) >= 0.5
    condyles['baseline_right']['mask'] = select_instance(condyles['baseline_right']['mask'])
    condyles['recall_right']['mask'] = ndimage.gaussian_filter(
        condyles['recall_right']['probs'], 1.2,
    ) >= 0.5
    condyles['recall_right']['mask'] = select_instance(condyles['recall_right']['mask'])

    return condyles


def main(in_dir, out_dir, verbose: bool=False):
    mandibles = infer_mandibles(in_dir, out_dir)
    save_volumes(out_dir, mandibles, 'mandible')

    condyles = infer_condyles(mandibles)
    save_volumes(out_dir, condyles, 'condyles')

    condyles = register_rami(mandibles, condyles, verbose=verbose)
    save_volumes(out_dir, condyles, 'matched')

    condyles = select_condyles(condyles)
    save_volumes(out_dir, condyles, 'cropped')

    interp_condyles = interpolate_condyles(in_dir, condyles)    
    save_volumes(out_dir, interp_condyles, 'interp')

    volumes = {}
    for moment in condyles:
        num_voxels = condyles[moment]['mask'].sum().item()
        volumes[moment] = num_voxels * 0.4**3
        print(f'{moment} (mm3): {num_voxels * 0.4**3:.0f}')

    return volumes


def validate(in_dir: Path, out_dir: Path, rmtree: bool=True):
    in_files = sorted(in_dir.glob('*.nii.gz'))
    try:
        assert len(in_files) == 2, 'Please provide exactly two nii.gz files.'
        assert 'baseline' in in_files[0].stem, 'Please use "baseline" in baseline scan filename.'
        assert 'recall' in in_files[1].stem, 'Please use "recall" in recall scan filename.'
    except Exception:
        return False

    if rmtree:
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
        return False
    
    return True


def process_case(out_dir: Path, mp: bool, in_case_dir: Path, verbose: bool=False):
    if not in_case_dir.is_dir():
        return None

    out_case_dir = out_dir / in_case_dir.name
    if not validate(
        in_case_dir,
        out_case_dir,
        rmtree=out_dir.as_posix() == '/output',
    ):
        return None

    try:
        if mp:
            os.chdir(Path(sys.path[-1]).parent.as_posix())
        volumes = main(in_case_dir, out_case_dir, verbose=verbose)
    except Exception as e:
        print(traceback.print_exception(e))
        print('Error:', in_case_dir)
        return None
    
    return in_case_dir.name, volumes


if __name__ == '__main__':
    mp.set_start_method('spawn')

    in_dir = Path('/mnt/diag/CBCT/condyle_resorption/Esbjerg/Cases')
    out_dir = Path('/mnt/diag/CBCT/condyle_resorption/Esbjerg/tmp')

    # in_dir = Path('/input')
    # out_dir = Path('/output')

    case_volumes = {}
    in_case_dirs = [d for d in sorted(in_dir.glob('*')) if d.is_dir()]
    in_case_dirs = [in_dir / 'Case 14']
    if False:
        with mp.Pool(4) as p:
            i = p.imap_unordered(partial(process_case, out_dir, True), in_case_dirs)
            t = tqdm(i, total=len(in_case_dirs))
            for out in t:
                if out is None:
                    continue

                case_name, volumes = out
                t.set_description(case_name)
                case_volumes[case_name] = volumes
    else:
        t = tqdm(in_case_dirs)
        for in_case_dir in t:
            out = process_case(out_dir, False, in_case_dir, verbose=True)
            if out is None:
                continue

            case_name, volumes = out
            t.set_description(case_name)
            case_volumes[case_name] = volumes

    with open(out_dir / 'volumes.json', 'w') as f:
        json.dump(case_volumes, f, indent=2)
