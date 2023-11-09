import copy
from pathlib import Path
from typing import Any, List

import nibabel
import numpy as np
from numpy.typing import NDArray
import pydicom
from tqdm import tqdm


class DICOM:

    def __init__(
        self,
        spacing: List[float],
        positions: List[float],
        orientation: List[float],
        content: NDArray[np.float32],
    ):
        self.spacing = spacing
        self.positions = positions
        self.orientation = orientation
        self.shape = content.shape
        self.content = content

    @property
    def affine(self) -> NDArray[np.float32]:
        F11, F21, F31 = self.orientation[3:]
        F12, F22, F32 = self.orientation[:3]

        slices = self.shape[0]

        dr, dc = self.spacing[:2]
        pos1 = self.positions[:3]
        posN = self.positions[3:]

        return np.array([
            [F11 * dr, F12 * dc, (posN[0] - pos1[0]) / (slices - 1), pos1[0]],
            [F21 * dr, F22 * dc, (posN[1] - pos1[1]) / (slices - 1), pos1[1]],
            [F31 * dr, F32 * dc, (posN[2] - pos1[2]) / (slices - 1), pos1[2]],
            [0, 0, 0, 1]
        ])
    
    def to_numpy(self) -> NDArray[np.int16]:
        return self.content.astype(np.int16)

    @staticmethod
    def rescale_intensities(ds: pydicom.Dataset) -> NDArray[Any]:
        intercept = getattr(ds, 'RescaleIntercept', None)
        slope = getattr(ds, 'RescaleSlope', None)

        if intercept is None or slope is None:
            de = ds[0x5200, 0x9229][0]
            de = de[0x0028, 0x9145][0]
            intercept = de.RescaleIntercept
            slope = de.RescaleSlope

        return ds.pixel_array * slope + intercept

    @staticmethod
    def voxel_spacing(ds: pydicom.Dataset) -> List[float]:
        in_plane = getattr(ds, 'PixelSpacing', None)
        slice_thickness = getattr(ds, 'SliceThickness', None)

        if in_plane is None or slice_thickness is None:
            ds = ds[0x5200, 0x9229][0]
            ds = ds[0x0028, 0x9110][0]

            in_plane = ds.PixelSpacing
            slice_thickness = ds.SliceThickness    

        spacing = [slice_thickness] + list(in_plane)
        spacing = spacing[::-1]
        spacing = [float(s) for s in spacing]

        return spacing

    @staticmethod
    def voxel_orientation(ds: pydicom.Dataset) -> List[float]:
        orientation = getattr(ds, 'ImageOrientationPatient', None)

        if orientation is None:
            slices = ds[0x5200, 0x9230]
            orientation = getattr(slices[0], 'ImageOrientationPatient', None)

        if orientation is None and (0x0020, 0x9116) in ds[0x5200, 0x9229][0]:
            ds = ds[0x5200, 0x9229][0]
            ds = ds[0x0020, 0x9116][0]
            orientation = getattr(ds, 'ImageOrientationPatient', None)

        if orientation is None and (0x0020, 0x9116) in slices[0]:
            de = slices[0][0x0020, 0x9116][0]
            orientation = de.ImageOrientationPatient

        orientation = [float(s) for s in orientation]

        return orientation

    @staticmethod
    def voxel_positions(ds: pydicom.Dataset) -> List[float]:
        pos1 = getattr(ds, 'ImagePositionPatient', None)
        posN = None

        if pos1 is None and (0x5200, 0x9230) in ds:
            slices = ds[0x5200, 0x9230]
            positions = []
            for idx in [0, -1]:
                if hasattr(slices[idx], 'ImagePositionPatient'):
                    position = slices[idx].ImagePositionPatient
                else:
                    ds = slices[idx][0x0020, 0x9113][0]
                    position = ds.ImagePositionPatient

                positions.append(position)

            pos1, posN = positions

        if pos1 is None:
            pos1 = [0, 0, 0]

        if posN is None:
            posN = copy.deepcopy(pos1)
            posN[2] -= (ds.NumberOfFrames - 1) * ds.SliceThickness
            
        positions = [float(c) for c in list(pos1) + list(posN)]

        return positions

    @staticmethod
    def read_files(path: Path):
        images, positions = [], []
        for dcm_file in sorted(path.glob('*')):
            ds = pydicom.dcmread(dcm_file)

            image = DICOM.rescale_intensities(ds)
            spacing = DICOM.voxel_spacing(ds)
            orientation = DICOM.voxel_orientation(ds)
            position = DICOM.voxel_positions(ds)

            images.append(image)
            positions.append(position)

        image = np.stack(images, axis=-1)[::-1, ::-1]
        positions = positions[0][:3] + positions[-1][3:]

        return DICOM(spacing, positions, orientation, image)
    

if __name__ == '__main__':
    root = Path('/mnt/diag/condyles/Esbjerg')

    in_dir = root / 'Data'
    out_dir = root / 'Cases'

    for dcm_dir in tqdm(sorted(in_dir.glob('*'))):
        if not dcm_dir.is_dir():
            continue

        print(dcm_dir)

        # dicom = DICOM.read_files(dcm_dir)

        case = dcm_dir.name.split('-')[0].strip()
        moment = dcm_dir.name.split('-')[-1].strip()
        out_file = out_dir / f'Case {case} - {moment}_0000.nii.gz'

        dicom_nii = nibabel.load(out_file)
        dicom = np.asarray(dicom_nii.dataobj)

        dicom = np.transpose(dicom, (1, 2, 0))[::-1, ::-1]

        out_file2 = out_dir / f'Case {case} - {moment}_0000.nii.gz'
        nii = nibabel.Nifti1Image(dicom, dicom_nii.affine)
        nibabel.save(nii, out_file2)
