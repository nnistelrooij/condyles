from time import perf_counter
from typing import Union

import numpy as np
import open3d
import SimpleITK as sitk
from skimage.measure import marching_cubes

reg = open3d.pipelines.registration

from register.utils import masked_marching_cubes


def ransac_icp(
    baseline,
    recall,
    voxel_size: float,
    verbose: bool=False,
    ransac: bool=False,
):
    count = perf_counter()

    # transform meshes to downsampled point clouds
    reference_pcd = open3d.geometry.PointCloud(
        open3d.utility.Vector3dVector(baseline),
    )
    if voxel_size[0] > 1:
        reference_pcd = reference_pcd.voxel_down_sample(voxel_size[0])
    reference_pcd.estimate_normals()

    test_pcd = open3d.geometry.PointCloud(
        open3d.utility.Vector3dVector(recall),
    )
    if voxel_size[1] > 1:
        test_pcd = test_pcd.voxel_down_sample(voxel_size[1])
    test_pcd.estimate_normals()

    if verbose:
        reference_pcd.paint_uniform_color([1, 0, 0])
        test_pcd.paint_uniform_color([0.5, 0.5, 0.5])
        open3d.visualization.draw_geometries([reference_pcd, test_pcd])

    # compute curvature point features
    if ransac:
        fpfh_cfg = {
            'search_param': open3d.geometry.KDTreeSearchParamHybrid(
                radius=5 * max(voxel_size), max_nn=100,
            ),
        }
        reference_fpfh = reg.compute_fpfh_feature(reference_pcd, **fpfh_cfg)
        test_fpfh = reg.compute_fpfh_feature(test_pcd, **fpfh_cfg)

        # register point clouds globally given feature correspondences
        ransac_cfg = {
            'mutual_filter': True,
            'max_correspondence_distance': 1.5 * max(voxel_size),
            'estimation_method': reg.TransformationEstimationPointToPoint(),
            'ransac_n': 4,
            'checkers': [
                reg.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                reg.CorrespondenceCheckerBasedOnDistance(1.5 * max(voxel_size)),
            ],
            'criteria': reg.RANSACConvergenceCriteria(4000000, 0.9999),
        }
        result = reg.registration_ransac_based_on_feature_matching(
            reference_pcd, test_pcd, reference_fpfh, test_fpfh, **ransac_cfg,
        )

    # register point clouds locally given coordinate correspondences
    icp_cfg = {
        'max_correspondence_distance': max(voxel_size),
        'init': result.transformation if ransac else np.eye(4),
        'estimation_method': reg.TransformationEstimationPointToPlane(),
        'criteria': reg.ICPConvergenceCriteria(max_iteration=3000),
    }
    result = reg.registration_icp(reference_pcd, test_pcd, **icp_cfg)

    if verbose:
        baseline_hom = np.column_stack((
            baseline, np.ones(baseline.shape[0]),
        ))
        baseline_t = baseline_hom @ result.transformation.T
        baseline_t[:, 0] += 200
        reference_pcd_t = open3d.geometry.PointCloud(
            open3d.utility.Vector3dVector(baseline_t[:, :3]),
        )
        reference_pcd_t.estimate_normals()
        reference_pcd_t.paint_uniform_color([1, 0, 0])

        recall_t = recall.copy()
        recall_t[:, 0] += 200
        test_pcd_t =  open3d.geometry.PointCloud(
            open3d.utility.Vector3dVector(recall_t),
        )
        test_pcd_t.estimate_normals()
        test_pcd_t.paint_uniform_color([0.5, 0.5, 0.5])

        open3d.visualization.draw_geometries([
            reference_pcd, test_pcd,
            reference_pcd_t, test_pcd_t,
        ])


    return result


def surface_registration(
    baseline,
    recall_ramus,
    recall_mandible,
    verbose=False,
):
    baseline_verts = masked_marching_cubes(baseline[:2])
    # recall_verts = masked_marching_cubes(recall[:2])
    recall_verts, _, _, _ = marching_cubes(recall_mandible)

    if verbose:
        recall_mesh = open3d.geometry.PointCloud(
            open3d.utility.Vector3dVector(recall_verts),
        )
        recall_mesh.estimate_normals()
        recall_mesh.paint_uniform_color([0.5, 0.5, 0.5])
        baseline_mesh = open3d.geometry.PointCloud(
            open3d.utility.Vector3dVector(baseline_verts),
        )
        baseline_mesh.estimate_normals()
        baseline_mesh.paint_uniform_color([1, 0, 0])
        condyle_verts = masked_marching_cubes(baseline[2:])
        condyle_mesh = open3d.geometry.PointCloud(
            open3d.utility.Vector3dVector(condyle_verts),
        )
        condyle_mesh.estimate_normals()
        condyle_mesh.paint_uniform_color([0, 0, 1])

        open3d.visualization.draw_geometries([recall_mesh, condyle_mesh, baseline_mesh])

    baseline_centroid = np.column_stack(np.nonzero(baseline[0])).mean(0)
    recall_centroid = np.column_stack(np.nonzero(recall_ramus[0])).mean(0)
    baseline_centered_verts = baseline_verts - baseline_centroid
    recall_centered_verts = recall_verts - recall_centroid

    if verbose:
        recall_mesh = open3d.geometry.PointCloud(
            open3d.utility.Vector3dVector(recall_centered_verts),
        )
        recall_mesh.estimate_normals()
        recall_mesh.paint_uniform_color([0.5, 0.5, 0.5])
        baseline_mesh = open3d.geometry.PointCloud(
            open3d.utility.Vector3dVector(baseline_centered_verts),
        )
        baseline_mesh.estimate_normals()
        baseline_mesh.paint_uniform_color([1, 0, 0])
        condyle_mesh = open3d.geometry.PointCloud(
            open3d.utility.Vector3dVector(condyle_verts - baseline_centroid),
        )
        condyle_mesh.estimate_normals()
        condyle_mesh.paint_uniform_color([0, 0, 1])

        open3d.visualization.draw_geometries([recall_mesh, condyle_mesh, baseline_mesh])

    fitnesses, noises = [], []
    for _ in range(10):
        noise = 2 * np.random.randn(3)
        affine = ransac_icp(baseline_centered_verts + noise, recall_centered_verts, voxel_size=[1, 1])

        fitnesses.append(affine.fitness)
        noises.append(noise)

    print('Fitness:', max(fitnesses))
    noise = noises[np.argmax(fitnesses)]
    affine = ransac_icp(baseline_centered_verts + noise, recall_centered_verts, voxel_size=[1, 1])

    T1 = np.eye(4)
    T1[:3, 3] = noise - baseline_centroid

    T2 = np.eye(4)
    T2[:3, 3] = recall_centroid

    T = T2 @ affine.transformation @ T1

    if verbose:
        recall_mesh = open3d.geometry.PointCloud(
            open3d.utility.Vector3dVector(recall_verts),
        )
        recall_mesh.estimate_normals()
        recall_mesh.paint_uniform_color([0.5, 0.5, 0.5])
        baseline_hom = np.column_stack((
            baseline_verts, np.ones(baseline_verts.shape[0]),
        ))
        baseline_t = baseline_hom @ T.T
        baseline_mesh = open3d.geometry.PointCloud(
            open3d.utility.Vector3dVector(baseline_t[:, :3]),
        )
        baseline_mesh.estimate_normals()
        baseline_mesh.paint_uniform_color([1, 0, 0])

        condyle_hom = np.column_stack((
            condyle_verts, np.ones(condyle_verts.shape[0]),
        ))
        condyle_t = condyle_hom @ T.T
        condyle_mesh = open3d.geometry.PointCloud(
            open3d.utility.Vector3dVector(condyle_t[:, :3]),
        )
        condyle_mesh.estimate_normals()
        condyle_mesh.paint_uniform_color([0, 0, 1])

        open3d.visualization.draw_geometries([recall_mesh, condyle_mesh, baseline_mesh])

    return T
    


def voxel_registration(
    baseline,
    recall,
    voxel_size: Union[list[float],float]=1.0,
    init_T=None,
):
    baseline_voxels = np.column_stack(np.nonzero(baseline)).astype(float)
    recall_voxels = np.column_stack(np.nonzero(recall)).astype(float)


    if init_T is None:
        baseline_centroid = baseline_voxels.mean(axis=0)
        recall_centroid = recall_voxels.mean(axis=0)
        baseline_voxels += recall_centroid - baseline_centroid
    else:
        baseline_hom = np.column_stack((
            baseline_voxels, np.ones(baseline_voxels.shape[0]),
        ))
        baseline_voxels = baseline_hom @ init_T.T
        baseline_voxels = baseline_voxels[:, :3]

        baseline_centroid = baseline_voxels.mean(axis=0)
        baseline_voxels -= baseline_centroid
        recall_voxels -= baseline_centroid

    if not isinstance(voxel_size, list):
        voxel_size = [voxel_size, voxel_size]

    affine = ransac_icp(baseline_voxels, recall_voxels, voxel_size)

    if init_T is None:
        translation = np.eye(4)
        translation[:3, 3] = recall_centroid - baseline_centroid

        affine = affine @ translation
    else:
        translation = np.eye(4)
        translation[:3, 3] = -baseline_centroid

        affine = np.linalg.inv(translation) @ affine @ translation @ init_T

    return affine
