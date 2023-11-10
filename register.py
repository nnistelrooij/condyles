from time import perf_counter

import numpy as np
import open3d
reg = open3d.pipelines.registration


def ransac_icp(
    baseline,
    recall,
    voxel_size: float,
    verbose: bool=False,
):
    count = perf_counter()

    # transform meshes to downsampled point clouds
    reference_pcd = open3d.geometry.PointCloud(
        open3d.utility.Vector3dVector(baseline),
    )
    reference_pcd = reference_pcd.voxel_down_sample(voxel_size)
    reference_pcd.estimate_normals()

    test_pcd = open3d.geometry.PointCloud(
        open3d.utility.Vector3dVector(recall),
    )
    test_pcd = test_pcd.voxel_down_sample(voxel_size)
    test_pcd.estimate_normals()

    if verbose:
        open3d.visualization.draw_geometries([reference_pcd, test_pcd])

    # compute curvature point features
    fpfh_cfg = {
        'search_param': open3d.geometry.KDTreeSearchParamHybrid(
            radius=5 * voxel_size, max_nn=100,
        ),
    }
    reference_fpfh = reg.compute_fpfh_feature(reference_pcd, **fpfh_cfg)
    test_fpfh = reg.compute_fpfh_feature(test_pcd, **fpfh_cfg)

    # register point clouds globally given feature correspondences
    ransac_cfg = {
        'mutual_filter': True,
        'max_correspondence_distance': 1.5 * voxel_size,
        'estimation_method': reg.TransformationEstimationPointToPoint(),
        'ransac_n': 4,
        'checkers': [
            reg.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            reg.CorrespondenceCheckerBasedOnDistance(1.5 * voxel_size),
        ],
        'criteria': reg.RANSACConvergenceCriteria(4000000, 0.9999),
    }
    result = reg.registration_ransac_based_on_feature_matching(
        reference_pcd, test_pcd, reference_fpfh, test_fpfh, **ransac_cfg,
    )

    # register point clouds locally given coordinate correspondences
    icp_cfg = {
        'max_correspondence_distance': voxel_size,
        'init': result.transformation,
        'estimation_method': reg.TransformationEstimationPointToPlane(),
        'criteria': reg.ICPConvergenceCriteria(max_iteration=3000),
    }
    result = reg.registration_icp(reference_pcd, test_pcd, **icp_cfg)

    print(f'registration time: {perf_counter() - count:.3f}s', )

    return result.transformation


def register(baseline, recall, voxel_size: float,):
    baseline_voxels = np.column_stack(np.nonzero(baseline))
    recall_voxels = np.column_stack(np.nonzero(recall))

    affine = ransac_icp(baseline_voxels, recall_voxels, voxel_size)

    return affine
