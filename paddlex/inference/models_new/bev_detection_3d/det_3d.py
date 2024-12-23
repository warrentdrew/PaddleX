# copyright (c) 2024 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Generic, List, Optional
import copy
import math

import numpy as np
import lazy_paddle as paddle


class _EasyDict(dict):
    def __getattr__(self, key: str):
        if key in self:
            return self[key]
        return super().__getattr__(self, key)

    def __setattr__(self, key: str, value: Generic):
        self[key] = value


class SampleMeta(_EasyDict):
    """ """

    # yapf: disable
    __slots__ = [
        "camera_intrinsic",
        # bgr or rgb
        "image_format",
        # pillow or cv2
        "image_reader",
        # chw or hwc
        "channel_order",
        # Unique ID of the sample
        "id",
        "time_lag",
        "ref_from_curr"
    ]
    # yapf: enable

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class Sample(_EasyDict):
    """ """

    _VALID_MODALITIES = ["image", "lidar", "radar", "multimodal", "multiview"]

    def __init__(self, path: str, modality: str):
        if modality not in self._VALID_MODALITIES:
            raise ValueError(
                "Only modality {} is supported, but got {}".format(
                    self._VALID_MODALITIES, modality
                )
            )

        self.meta = SampleMeta()

        self.path = path
        self.data = None
        self.modality = modality.lower()

        self.bboxes_2d = None
        self.bboxes_3d = None
        self.labels = None

        self.sweeps = []
        self.attrs = None


class NormalCustom(paddle.distribution.Normal):
    def __init__(self, loc, scale, name=None):
        super(NormalCustom, self).__init__(loc, scale, name)

    def cdf(self, value):
        return 0.5 * (
            1 + paddle.erf((value - self.loc) * self.scale.reciprocal() / math.sqrt(2))
        )


def translate(points: np.ndarray, x: np.ndarray) -> None:
    """
    Applies a translation to the point cloud.
    :param x: <np.float: 3, 1>. Translation in x, y, z.
    """
    for i in range(3):
        points[i, :] = points[i, :] + x[i]


def rotate(points: np.ndarray, rot_matrix: np.ndarray) -> None:
    """
    Applies a rotation.
    :param rot_matrix: <np.float: 3, 3>. Rotation matrix.
    """
    points[:3, :] = np.dot(rot_matrix, points[:3, :])


def view_points(points: np.ndarray, view: np.ndarray, normalize: bool) -> np.ndarray:

    assert view.shape[0] <= 4
    assert view.shape[1] <= 4
    assert points.shape[0] == 3

    viewpad = np.eye(4)
    viewpad[: view.shape[0], : view.shape[1]] = view

    nbr_points = points.shape[1]

    # Do operation in homogenous coordinates.
    points = np.concatenate((points, np.ones((1, nbr_points))))
    points = np.dot(viewpad, points)
    points = points[:3, :]

    if normalize:
        # TODO: check why have zeros value here
        z = points[2:3, :].repeat(3, 0).reshape(3, nbr_points)
        points = points / (z + 1e-10)

    return points


def view_points(points: np.ndarray, view: np.ndarray, normalize: bool) -> np.ndarray:

    assert view.shape[0] <= 4
    assert view.shape[1] <= 4
    assert points.shape[0] == 3

    viewpad = np.eye(4)
    viewpad[: view.shape[0], : view.shape[1]] = view

    nbr_points = points.shape[1]

    # Do operation in homogenous coordinates.
    points = np.concatenate((points, np.ones((1, nbr_points))))
    points = np.dot(viewpad, points)
    points = points[:3, :]

    if normalize:
        # TODO: check why have zeros value here
        z = points[2:3, :].repeat(3, 0).reshape(3, nbr_points)
        points = points / (z + 1e-10)

    return points


def map_pointcloud_to_image(
    points, img, sensor2lidar_r, sensor2lidar_t, camera_intrinsic, show=False
):
    """Project points to image plane
    This code is based on https://github.com/ADLab-AutoDrive/BEVFusion/blob/3f992837ad659f050df38d7b0978372425be16ff/mmdet3d/core/visualizer/image_vis.py#L91
    """
    points = copy.deepcopy(points.T)
    translate(points, -sensor2lidar_t)
    rotate(points, sensor2lidar_r.T)

    # Fifth step: actually take a "picture" of the point cloud.
    # Grab the depths (camera frame z axis points away from the camera).
    depths = points[2, :]
    coloring = depths
    # Take the actual picture (matrix multiplication with camera-matrix + renormalization).
    points = view_points(points[:3, :], camera_intrinsic, normalize=True)

    # Remove points that are either outside or behind the camera. Leave a margin of 1 pixel for aesthetic reasons.
    # Also make sure points are at least 1m in front of the camera to avoid seeing the lidar points on the camera
    # casing for non-keyframes which are slightly out of sync.
    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > 1.0)
    mask = np.logical_and(mask, points[0, :] > 1)
    mask = np.logical_and(mask, points[0, :] < img.shape[1] - 1)
    mask = np.logical_and(mask, points[1, :] > 1)
    mask = np.logical_and(mask, points[1, :] < img.shape[0] - 1)
    points = points[:, mask]
    coloring = coloring[mask]
    depth_map = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
    xs = np.minimum((points[0, :] + 0.5).astype(np.int32), depth_map.shape[1])
    ys = np.minimum((points[1, :] + 0.5).astype(np.int32), depth_map.shape[0])
    for x, y, c in zip(xs, ys, coloring):
        depth_map[y, x] = c
    if show:
        raise NotImplementedError
    return depth_map
