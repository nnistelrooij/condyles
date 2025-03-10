from pathlib import Path

import matplotlib.pyplot as plt
import nibabel
import numpy as np
from scipy import ndimage

import os, sys
sys.path.append(str(Path(os.getcwd()) / 'jawfrac'))
os.chdir(sys.path[-1])

from jawfrac.visualization import visualize


def wolla(path, threshold: int, min_voxels: int, dilate: int=0, color: str='double'):
    img = nibabel.load(path)

    data = np.asarray(img.dataobj)
    affine = img.affine
    mask = data >= threshold

    if dilate:
        mask = ndimage.binary_dilation(
            input=mask,
            structure=ndimage.generate_binary_structure(3, 1),
            iterations=dilate,
        )

    labels, _ = ndimage.label(mask, ndimage.generate_binary_structure(3, 1))
    _, inverse, counts = np.unique(labels.flatten(), return_inverse=True, return_counts=True)
    print(np.sort(counts))

    labels[(counts < min_voxels)[inverse].reshape(labels.shape)] = 0
    labels = labels[::-1]

    points = np.column_stack(labels.nonzero()).astype(float)
    hom_points = np.column_stack((points, np.ones_like(points[:, 0])))
    points = np.einsum('ij,kj->ki', affine, hom_points)
    points = points[:, :-1]

    if color == 'double':
        colors = points[:, 0] - points[:, 0].mean()
        colors[colors > 0] = (colors[colors > 0] - colors[colors > 0].min()) / (colors[colors > 0].max() - colors[colors > 0].min())
        colors[colors < 0] = 1 - (colors[colors < 0] - colors[colors < 0].min()) / (colors[colors < 0].max() - colors[colors < 0].min())
    elif color == 'single':
        colors = points[:, 0] - points[:, 0].min()
        colors = (colors - colors.min()) / (colors.max() - colors.min())

    # colors = np.abs(colors - colors.mean())
    # colors = (colors - colors.min()) / (colors.max() - colors.min())
    colors = np.tile(colors, (3, 1)).T

    return labels, points, colors



def wolla2(path, path2, threshold: int, min_voxels: int, dilate: int=0, color: str='double'):
    img1 = nibabel.load(path)
    img2 = nibabel.load(path2)

    data1 = np.asarray(img1.dataobj)
    data2 = np.asarray(img2.dataobj)
    affine = img1.affine
    mask = (data1 >= threshold) | (data2 >= threshold)

    if dilate:
        mask = ndimage.binary_dilation(
            input=mask,
            structure=ndimage.generate_binary_structure(3, 1),
            iterations=dilate,
        )

    labels, _ = ndimage.label(mask, ndimage.generate_binary_structure(3, 1))
    _, inverse, counts = np.unique(labels.flatten(), return_inverse=True, return_counts=True)

    labels[(counts < min_voxels)[inverse].reshape(labels.shape)] = 0
    labels = labels[::-1]

    points = np.column_stack(labels.nonzero()).astype(float)
    hom_points = np.column_stack((points, np.ones_like(points[:, 0])))
    points = np.einsum('ij,kj->ki', affine, hom_points)
    points = points[:, :-1]

    if color == 'double':
        colors = points[:, 0] - points[:, 0].mean()
        colors[colors > 0] = (colors[colors > 0] - colors[colors > 0].min()) / (colors[colors > 0].max() - colors[colors > 0].min())
        colors[colors < 0] = 1 - (colors[colors < 0] - colors[colors < 0].min()) / (colors[colors < 0].max() - colors[colors < 0].min())
    elif color == 'single':
        colors = points[:, 0] - points[:, 0].min()
        colors = (colors - colors.min()) / (colors.max() - colors.min())

    # colors = np.abs(colors - colors.mean())
    # colors = (colors - colors.min()) / (colors.max() - colors.min())
    colors = np.tile(colors, (3, 1)).T

    return labels, points, colors


def visualize_input(path):
    _, points, colors = wolla(path, 300, 20_000)
    points[:, 0] *= -1

    visualize(points, colors * 255)


def visualize_mandible(path, path2):
    labels, points, colors = wolla(path, 400, 20_000)
    # points[:, 0] *= -1

    mandible_labels, mandible_points, mandible_colors = wolla(path2, 1, 1000, 1)
    # mandible_points[:, 0] *= -1

    # mandible_colors[:, 0] = 1
    mandible_colors[:, 0] = mandible_colors[:, 1]
    mandible_colors[:, 1] = 0
    mandible_colors[:, 2] = 0

    both_labels = np.concatenate((
        np.column_stack(labels.nonzero()),
        np.column_stack(mandible_labels.nonzero()),
    ))
    both_labels, inverse, counts = np.unique(both_labels, axis=0, return_inverse=True, return_counts=True)

    labels = np.concatenate((
        labels[(counts == 1)[inverse[:labels.shape[0]]]],
        mandible_labels,
    ))
    points = np.concatenate((
        points[(counts == 1)[inverse[:points.shape[0]]]],
        mandible_points,
    ))
    colors = np.concatenate((
        colors[(counts == 1)[inverse[:colors.shape[0]]]],
        mandible_colors,
    ))

    return labels, points, colors


def visualize_union(path, path2, path3):
    labels, points, colors = wolla(path, 400, 20_000)
    # points[:, 0] *= -1

    mandible_labels, mandible_points, mandible_colors = wolla2(path2, path3, 1, 1000, 1)
    # mandible_points[:, 0] *= -1

    # mandible_colors[:, 0] = 1
    mandible_colors[:, 0] = mandible_colors[:, 1]
    mandible_colors[:, 1] = 0
    mandible_colors[:, 2] = 0

    both_labels = np.concatenate((
        np.column_stack(labels.nonzero()),
        np.column_stack(mandible_labels.nonzero()),
    ))
    both_labels, inverse, counts = np.unique(both_labels, axis=0, return_inverse=True, return_counts=True)

    labels = np.concatenate((
        labels[(counts == 1)[inverse[:labels.shape[0]]]],
        mandible_labels,
    ))
    points = np.concatenate((
        points[(counts == 1)[inverse[:points.shape[0]]]],
        mandible_points,
    ))
    colors = np.concatenate((
        colors[(counts == 1)[inverse[:colors.shape[0]]]],
        mandible_colors,
    ))

    return labels, points, colors

def visualize_mandible_fracture(path, path2, path3):
    labels, points, colors = visualize_mandible(path, path2)

    mandible_labels, mandible_points, mandible_colors = wolla(path3, 1, 1000, 0)
    # mandible_points[:, 0] *= -1

    # mandible_colors[:, 0] = 1
    mandible_colors[:, 0] = 1 - mandible_colors[:, 0]
    mandible_colors[:, 1:] = 0

    both_labels = np.concatenate((
        np.column_stack(labels.nonzero()),
        np.column_stack(mandible_labels.nonzero()),
    ))
    both_labels, inverse, counts = np.unique(both_labels, axis=0, return_inverse=True, return_counts=True)

    points = np.concatenate((
        points[(counts == 1)[inverse[:points.shape[0]]]],
        mandible_points,
    ))
    colors = np.concatenate((
        colors[(counts == 1)[inverse[:colors.shape[0]]]],
        mandible_colors,
    ))

    visualize(points, colors * 255)


def visualize_patch(path: Path):
    img = nibabel.load(path)

    data = np.asarray(img.dataobj)

    data = ndimage.zoom(data, [1, 1, 2.5])

    img = np.full((154, 154), 255)
    for i, idx in enumerate(range(188, 159, -4)):
        img[
            8 * i:98 + 8 * i,
            56 - 8 * i:154 - 8 * i,
        ] = np.pad(data[idx, 209:305, 246:342], (1, 1))
    
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.savefig(r'C:\Users\Niels-laptop\Documents\Master Thesis\patch.png', bbox_inches='tight', pad_inches=0)
    plt.show()


def visualize_mandible_patch(path: Path, path2,):
    img = nibabel.load(path)

    data = np.asarray(img.dataobj)

    data = ndimage.zoom(data, [1, 1, 2.5])
    
    mandible = nibabel.load(path2)

    mandible = np.asarray(mandible.dataobj)
    mandible = ndimage.zoom(mandible, [1, 1, 2.5], output=float).round().astype(bool)

    data = data.repeat(3).reshape(data.shape + (3,)).astype(float)
    mandible = mandible.nonzero()
    mandible = np.concatenate((
        np.column_stack(mandible + (np.full(mandible[0].shape[:1], 1),)),
        np.column_stack(mandible + (np.full(mandible[0].shape[:1], 2),)),
    ))
    data[tuple(mandible.T)] *= 0.5

    out = np.full((154, 154, 3), 255)
    for i, idx in enumerate(range(188, 159, -4)):
        out[
            8 * i:98 + 8 * i,
            56 - 8 * i:154 - 8 * i,
        ] = np.pad(data[idx, 209:305, 246:342], ((1, 1), (1, 1), (0, 0)))
    
    plt.imshow(out, cmap='gray')
    plt.axis('off')
    plt.savefig(r'C:\Users\Niels-laptop\Documents\Master Thesis\mandible_patch.png', bbox_inches='tight', pad_inches=0)
    plt.show()


def visualize_expand_label(scan_path, frac_path):
    img = nibabel.load(scan_path)
    shape = np.array(img.header.get_data_shape())
    intensities = np.asarray(img.dataobj)
    if intensities.min() == 0 and intensities.max() == 255:
        intensities = (intensities - intensities[intensities > 0].mean()) / 255 * 4096

    print('start')
    intensities = ndimage.zoom(
        input=intensities,
        zoom=np.array(img.header.get_zooms()) / 0.4,
    )
    print('interp 1')


    img = nibabel.load(frac_path)
    lbl = np.asarray(img.dataobj)
    label = ndimage.zoom(
        input=lbl,
        zoom=np.array(img.header.get_zooms()) / 0.4,
        output=float,
    ).round().astype(bool)
    print('interp 2')

    out = ndimage.binary_dilation(
        input=label,
        structure=ndimage.generate_binary_structure(3, 2),
        iterations=1,
        mask=intensities >= 300
    )
    out = ndimage.binary_dilation(
        input=out,
        structure=ndimage.generate_binary_structure(3, 2),
        iterations=1,
    ).astype(np.float32)

    out = ndimage.gaussian_filter(
        input=out.astype(np.float32),
        sigma=0.5,
    )

    out = ndimage.zoom(
        input=out,
        zoom=0.4 / np.array(img.header.get_zooms()),
    ) >= 0.1


    # out = out.clip(0, 1)
    # out = np.tile(out[:, :, 59], (3, 1, 1)).transpose(1, 2, 0)
    # out[:, :, 1:] = 0
    # plt.imshow(out)
    # plt.axis('off')
    # plt.savefig('/mnt/d/Users/Niels-laptop/Documents/Master Thesis/expand_label_3.png', bbox_inches='tight', pad_inches=0)
    # plt.show()

    out = out[:shape[0], :shape[1], :shape[2]].astype(np.uint16)

    lbl[(lbl == 0) & (out == 1)] = 3

    img = nibabel.Nifti1Image(lbl, img.affine)
    nibabel.save(img, scan_path.parent / 'expand_label.nii.gz')



if __name__ == '__main__':
    # Case 16, 20

    path = Path('/mnt/diag/CBCT/condyle_resorption/Esbjerg/Results')
    case = 'Case 19'  # 16, 19

    labels, points, colors = wolla(path / case / 'baseline_cbct.nii.gz', 400, 20_000)
    visualize(points, 255 * colors)

    labels, points, colors = visualize_mandible(
        path / case / 'baseline_cbct.nii.gz',
        path / case / 'baseline_seg_mandible.nii.gz',
    )
    visualize(points, 255 * colors)

    for is_left in [True, False]:
        labels, points, colors = wolla(f'/home/mkaailab/Documents/CBCT/condyles/nnUNet_raw/Dataset008_EsbjergResultsCondyles/imagesTr/condyle_{case[-2:]}0{int(is_left)}_0000.nii.gz', 400, 20_000)
        visualize(points, 255 * colors)

        labels, points, colors = visualize_mandible(
            path / case / f'/home/mkaailab/Documents/CBCT/condyles/nnUNet_raw/Dataset008_EsbjergResultsCondyles/imagesTr/condyle_{case[-2:]}0{int(is_left)}_0000.nii.gz',
            path / case / f'/home/mkaailab/Documents/CBCT/condyles/nnUNet_raw/Dataset008_EsbjergResultsCondyles/labelsTr/condyle_{case[-2:]}0{int(is_left)}.nii.gz',
        )
        visualize(points, 255 * colors)

    labels, points, colors = visualize_union(
        path / case / 'baseline_cbct.nii.gz',
        path / case / 'baseline_seg_mandible.nii.gz',
        path / case / 'baseline_seg_condyles.nii.gz',
    )
    visualize(points, 255 * colors)
    
    labels, points, colors = visualize_mandible(
        path / case / 'recall_cbct.nii.gz',
        path / case / 'recall_condyles_matched.nii.gz',
    )
    visualize(points, 255 * colors)

    # visualize_patch(path)

    visualize_mandible(path, path.parent / 'mandible.nii.gz')
    # visualize_mandible_patch(path)

    # visualize_mandible_patch(path, path.parent / 'frac_pred_linear.nii.gz')

    visualize_expand_label(
        path / '81' / 'Patient81_main_image.nii.gz',
        path / '81' / 'label.nii.gz',
    )
