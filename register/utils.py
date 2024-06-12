from typing import Literal

import numpy as np
import open3d
from scipy import ndimage
from skimage.measure import marching_cubes


def transform(baseline, recall, affine, method: Literal['slow', 'fast']='slow'):
    if method == 'slow':
        out = ndimage.affine_transform(
            input=baseline.astype(np.float32),
            matrix=np.linalg.inv(affine),
            output_shape=recall.shape,
        )
        return out

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


def apply_translation(affine, translation1, translation2):
    affine_translate1 = np.eye(4)
    affine_translate1[:3, 3] = [slc.start for slc in translation1]
    out = np.linalg.inv(affine_translate1) @ affine @ affine_translate1

    affine_translate2 = np.eye(4)
    affine_translate2[:3, 3] = [slc.start for slc in translation2]
    out = out @ affine_translate1 @ np.linalg.inv(affine_translate2)

    return out


def masked_marching_cubes(masks, verbose=False):
    pos_verts, pos_faces, _, _ = marching_cubes(masks[0])
    mask_verts, mask_faces, _, _ = marching_cubes(masks[1])

    mesh = open3d.geometry.TriangleMesh(
        open3d.utility.Vector3dVector(mask_verts),
        open3d.utility.Vector3iVector(mask_faces),
    )

    if verbose:
        mesh.compute_vertex_normals()
        pos_mesh = open3d.geometry.TriangleMesh(
            open3d.utility.Vector3dVector(pos_verts),
            open3d.utility.Vector3iVector(pos_faces),
        )
        pos_mesh.compute_vertex_normals()
        open3d.visualization.draw_geometries([mesh, pos_mesh])

    scene = open3d.t.geometry.RaycastingScene()
    scene.add_triangles(open3d.t.geometry.TriangleMesh.from_legacy(mesh))

    dists = scene.compute_distance(pos_verts).numpy()

    pos_verts = pos_verts[dists > 1]

    return pos_verts
