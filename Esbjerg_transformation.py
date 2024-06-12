import copy
from pathlib import Path

import multiprocessing as mp
import nibabel
import numpy as np
import open3d
import open3d.visualization
import pymeshlab
from scipy import ndimage
from tqdm import tqdm

from register.voxel_based import ransac_icp


def mandibular_mask(root):
    for moment in ['preop', 'postop']:
        mask_nii = nibabel.load(root / f'{moment}_mandible.nii.gz')
        try:
            mask = np.asarray(mask_nii.dataobj)
        except Exception as e:
            print(root)
            raise e
        counts = ndimage.sum_labels(np.ones_like(mask), mask, range(mask.max() + 1))

        fg_intensity = counts.argmax()

        mask = mask == fg_intensity

        affine = mask_nii.affine
        orientation = nibabel.io_orientation(affine)

        mask = nibabel.apply_orientation(mask, orientation)

        mask_nii = nibabel.Nifti1Image(mask.astype(np.uint16), np.eye(4))
        nibabel.save(mask_nii, root / f'{moment}_mask.nii.gz')


def load_renders(root: Path, moment: str):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(root / f'{moment}.stl'))
    vertices = ms.current_mesh().vertex_matrix()
    triangles = ms.current_mesh().face_matrix()

    if 'Case11' in root.as_posix() or 'Case17' in root.as_posix() or 'Case02' in root.as_posix():
        z_min = np.quantile(vertices[:, 2], 0.85)  # 0.85 bad, 0.90 bad
    else:
        z_min = np.quantile(vertices[:, 2], 0.9)  # 0.85 bad, 0.90 bad
    y_max = np.quantile(vertices[:, 1], 0.25)
    left_x_max = np.quantile(vertices[:, 0], 0.2)
    right_x_min = np.quantile(vertices[:, 0], 0.8)

    left_mask = (
        (vertices[:, 0] <= left_x_max)
        & (vertices[:, 1] <= y_max)
        & (vertices[:, 2] >= z_min)
    )
    right_mask = (
        (vertices[:, 0] >= right_x_min)
        & (vertices[:, 1] <= y_max)
        & (vertices[:, 2] >= z_min)
    )

    left_vertices = vertices[left_mask]
    vertex_map = np.cumsum(left_mask) - 1
    left_triangles = vertex_map[triangles[np.all(left_mask[triangles], axis=1)]]

    right_vertices = vertices[right_mask]
    vertex_map = np.cumsum(right_mask) - 1
    right_triangles = vertex_map[triangles[np.all(right_mask[triangles], axis=1)]]

    left_condyle = open3d.geometry.TriangleMesh(
        open3d.utility.Vector3dVector(left_vertices),
        open3d.utility.Vector3iVector(left_triangles)
    )
    left_condyle.compute_vertex_normals()
    right_condyle = open3d.geometry.TriangleMesh(
        open3d.utility.Vector3dVector(right_vertices),
        open3d.utility.Vector3iVector(right_triangles)
    )
    right_condyle.compute_vertex_normals()

    return left_condyle, right_condyle


def transform(reference, test, spacings, verbose: bool=False):
    # scale
    # T1 = np.eye(4)
    # T1[np.diag_indices(3)] = 1 / spacings
    # inter = copy.deepcopy(reference).transform(T1)

    # rotate
    T2 = np.eye(4)
    T2[0, 0] = -1
    T2[1, 1] = -1
    inter = copy.deepcopy(reference).transform(T2)

    # translate
    inter_vertices = np.asarray(inter.vertices)
    z_cutoff = np.quantile(inter_vertices[:, 2], 0.95)
    inter_keypoint = inter_vertices[inter_vertices[:, 2] >= z_cutoff].mean(0)

    test_vertices = np.asarray(test.vertices)
    z_cutoff = np.quantile(test_vertices[:, 2], 0.9)
    test_keypoint = test_vertices[test_vertices[:, 2] >= z_cutoff].mean(0)

    T3 = np.eye(4)
    T3[:3, 3] = test_keypoint - inter_keypoint
    inter = inter.transform(T3)

    if verbose:
        open3d.visualization.draw_geometries([inter, test])

    # register
    affine = ransac_icp(
        np.asarray(inter.vertices),
        np.asarray(test.vertices),
        [1, 1],
        ransac=False,
    )
    print(affine)
    inter = inter.transform(affine.transformation)


    # complete affine matrix
    T = affine.transformation @ T3 @ T2  # @ T1
    reference = copy.deepcopy(reference).transform(T)
    
    if verbose:
        open3d.visualization.draw_geometries([inter, test, reference])

    return T



# 1.0 smoothing for 3D rendering most similar

if __name__ == '__main__':
    root = Path('/mnt/diag/CBCT/condyle_resorption/Esbjerg')
    case_dirs = list(sorted(root.glob('GroundTruth/Case*')))
    # with mp.Pool(10) as p:
    #     for _ in tqdm(p.imap_unordered(mandibular_mask, case_dirs), total=len(case_dirs)):
    #         pass
    # exit()
    for case_dir in case_dirs:
        case_name = ' '.join([case_dir.name[:-2], case_dir.name[-2:]])
        print(case_name)

        left_render, right_render = load_renders(
            case_dir, 'recall',
        )

        recall_path = root / 'Cases' / case_name / 'recall.nii.gz'
        recall_nii = nibabel.load(recall_path)
        spacings = np.array(recall_nii.header.get_zooms())


        for direction, render in {'left': left_render, 'right': right_render}.items():
            mimics_baseline = open3d.io.read_triangle_mesh(
                str(case_dir / f'preop {direction} condyle.stl')
            )
            mimics_recall = open3d.io.read_triangle_mesh(
                str(case_dir / f'postop {direction} condyle.stl')
            )
            mimics_baseline.compute_vertex_normals()
            mimics_recall.compute_vertex_normals()

            T = transform(mimics_recall, render, spacings, verbose=False)
            mimics_baseline = mimics_baseline.transform(T)
            mimics_recall = mimics_recall.transform(T)

            

            open3d.io.write_triangle_mesh(
                str(case_dir / f'preop {direction} condyle transformed.stl'),
                mimics_baseline,
            )
            open3d.io.write_triangle_mesh(
                str(case_dir / f'postop {direction} condyle transformed.stl'),
                mimics_recall,
            )
