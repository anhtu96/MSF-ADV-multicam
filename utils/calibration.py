import pandas as pd
from scipy.spatial.transform import Rotation
import numpy as np


def quat_to_mat(quat_wxyz):
    """Convert a quaternion to a 3D rotation matrix.

    NOTE: SciPy uses the scalar last quaternion notation. Throughout this repository,
        we use the scalar FIRST convention.

    Args:
        quat_wxyz: (...,4) array of quaternions in scalar first order.

    Returns:
        (...,3,3) 3D rotation matrix.
    """
    # Convert quaternion from scalar first to scalar last.
    quat_xyzw = quat_wxyz[..., [1, 2, 3, 0]]
    mat = Rotation.from_quat(quat_xyzw).as_matrix()
    return mat

def cart2hom(pts_3d):
    """ Input: nx3 points in Cartesian
        Oupput: nx4 points in Homogeneous by pending 1
    """
    n = pts_3d.shape[0]
    pts_3d_hom = np.hstack((pts_3d, np.ones((n, 1))))
    return pts_3d_hom

def get_camera_intrinsic_matrix(path):
    """Load camera calibration data and constructs intrinsic matrix.
    Args:
       camera_config: Calibration config in json
    Returns:
       Camera intrinsic matrix.
    """
    ft = pd.read_feather(path)
    K = []
    for cam in range(ft.shape[0]):
        intrinsic_matrix = np.eye(3, dtype=float)
        intrinsic_matrix[0, 0] = ft.loc[cam, "fx_px"]
        intrinsic_matrix[1, 1] = ft.loc[cam, "fy_px"]
        intrinsic_matrix[0, 2] = ft.loc[cam, "cx_px"]
        intrinsic_matrix[1, 2] = ft.loc[cam, "cy_px"]
        K.append(intrinsic_matrix)
    return K

def read_from_feather(path):
    calib_txt = pd.read_feather(path)
    rotations = quat_to_mat(calib_txt.loc[:, ["qw", "qx", "qy", "qz"]].to_numpy())
    translations = calib_txt.loc[:, ["tx_m", "ty_m", "tz_m"]].to_numpy()
    sensor_names = calib_txt.loc[:, "sensor_name"].to_numpy()
    return rotations, translations, sensor_names

def create_transform_matrix(rotation_mat, translation_mat):
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rotation_mat
    transform_matrix[:3, 3] = translation_mat
    return transform_matrix