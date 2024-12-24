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


import numbers
import cv2
import numpy as np
from typing import Generic, List, Optional
import lazy_paddle as paddle

from ...utils.io import ImageReader
from ....utils import logging
from ...common.reader.det_3d_reader import Sample


cv2_interp_codes = {
    "nearest": cv2.INTER_NEAREST,
    "bilinear": cv2.INTER_LINEAR,
    "bicubic": cv2.INTER_CUBIC,
    "area": cv2.INTER_AREA,
    "lanczos": cv2.INTER_LANCZOS4,
}


class LoadPointsFromFile:
    """Load Points From File."""

    def __init__(
        self, load_dim=6, use_dim=[0, 1, 2], shift_height=False, use_color=False
    ):
        self.shift_height = shift_height
        self.use_color = use_color
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert (
            max(use_dim) < load_dim
        ), f"Expect all used dimensions < {load_dim}, got {use_dim}"

        self.load_dim = load_dim
        self.use_dim = use_dim

    def _load_points(self, pts_filename):
        """Private function to load point clouds data."""

        points = np.fromfile(pts_filename, dtype=np.float32)
        return points

    def __call__(self, results):
        """Call function to load points data from file."""
        pts_filename = results["pts_filename"]
        points = self._load_points(pts_filename)
        points = points.reshape(-1, self.load_dim)
        points = points[:, self.use_dim]
        attribute_dims = None

        if self.shift_height:
            floor_height = np.percentile(points[:, 2], 0.99)
            height = points[:, 2] - floor_height
            points = np.concatenate(
                [points[:, :3], np.expand_dims(height, 1), points[:, 3:]], 1
            )
            attribute_dims = dict(height=3)

        if self.use_color:
            assert len(self.use_dim) >= 6
            if attribute_dims is None:
                attribute_dims = dict()
            attribute_dims.update(
                dict(
                    color=[
                        points.shape[1] - 3,
                        points.shape[1] - 2,
                        points.shape[1] - 1,
                    ]
                )
            )

        results["points"] = points

        return results


class LoadPointsFromMultiSweeps(object):
    """Load points from multiple sweeps.

    This is usually used for nuScenes dataset to utilize previous sweeps.

    Args:
        sweeps_num (int): Number of sweeps. Defaults to 10.
        load_dim (int): Dimension number of the loaded points. Defaults to 5.
        use_dim (list[int]): Which dimension to use. Defaults to [0, 1, 2, 4].
            for more details. Defaults to dict(backend='disk').
        pad_empty_sweeps (bool): Whether to repeat keyframe when
            sweeps is empty. Defaults to False.
        remove_close (bool): Whether to remove close points.
            Defaults to False.
        test_mode (bool): If test_model=True used for testing, it will not
            randomly sample sweeps but select the nearest N frames.
            Defaults to False.
    """

    def __init__(
        self,
        sweeps_num=10,
        load_dim=5,
        use_dim=[0, 1, 2, 4],
        pad_empty_sweeps=False,
        remove_close=False,
        test_mode=False,
        point_cloud_angle_range=None,
    ):
        self.load_dim = load_dim
        self.sweeps_num = sweeps_num
        self.use_dim = use_dim
        self.pad_empty_sweeps = pad_empty_sweeps
        self.remove_close = remove_close
        self.test_mode = test_mode

        if point_cloud_angle_range is not None:
            self.filter_by_angle = True
            self.point_cloud_angle_range = point_cloud_angle_range
            print(point_cloud_angle_range)
        else:
            self.filter_by_angle = False
            # self.point_cloud_angle_range = point_cloud_angle_range

    def _load_points(self, pts_filename):
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        points = np.fromfile(pts_filename, dtype=np.float32)
        return points

    def _remove_close(self, points, radius=1.0):
        """Removes point too close within a certain radius from origin.

        Args:
            points (np.ndarray): Sweep points.
            radius (float): Radius below which points are removed.
                Defaults to 1.0.

        Returns:
            np.ndarray: Points after removing.
        """
        if isinstance(points, np.ndarray):
            points_numpy = points
        else:
            raise NotImplementedError
        x_filt = np.abs(points_numpy[:, 0]) < radius
        y_filt = np.abs(points_numpy[:, 1]) < radius
        not_close = np.logical_not(np.logical_and(x_filt, y_filt))
        return points[not_close]

    def filter_point_by_angle(self, points):
        if isinstance(points, np.ndarray):
            points_numpy = points
        else:
            raise NotImplementedError
        pts_phi = (
            np.arctan(points_numpy[:, 0] / points_numpy[:, 1])
            + (points_numpy[:, 1] < 0) * np.pi
            + np.pi * 2
        ) % (np.pi * 2)

        pts_phi[pts_phi > np.pi] -= np.pi * 2
        pts_phi = pts_phi / np.pi * 180

        assert np.all(-180 <= pts_phi) and np.all(pts_phi <= 180)

        filt = np.logical_and(
            pts_phi >= self.point_cloud_angle_range[0],
            pts_phi <= self.point_cloud_angle_range[1],
        )
        return points[filt]

    def __call__(self, results):
        """Call function to load multi-sweep point clouds from files.

        Args:
            results (dict): Result dict containing multi-sweep point cloud \
                filenames.

        Returns:
            dict: The result dict containing the multi-sweep points data. \
                Added key and value are described below.

                - points (np.ndarray): Multi-sweep point cloud arrays.
        """
        points = results["points"]
        points[:, 4] = 0
        sweep_points_list = [points]
        ts = results["timestamp"]
        if self.pad_empty_sweeps and len(results["sweeps"]) == 0:
            for i in range(self.sweeps_num):
                if self.remove_close:
                    sweep_points_list.append(self._remove_close(points))
                else:
                    sweep_points_list.append(points)
        else:
            if len(results["sweeps"]) <= self.sweeps_num:
                choices = np.arange(len(results["sweeps"]))
            elif self.test_mode:
                choices = np.arange(self.sweeps_num)
            else:
                choices = np.random.choice(
                    len(results["sweeps"]), self.sweeps_num, replace=False
                )
            for idx in choices:
                sweep = results["sweeps"][idx]
                points_sweep = self._load_points(sweep["data_path"])
                points_sweep = np.copy(points_sweep).reshape(-1, self.load_dim)
                if self.remove_close:
                    points_sweep = self._remove_close(points_sweep)
                sweep_ts = sweep["timestamp"] / 1e6
                points_sweep[:, :3] = (
                    points_sweep[:, :3] @ sweep["sensor2lidar_rotation"].T
                )
                points_sweep[:, :3] += sweep["sensor2lidar_translation"]
                points_sweep[:, 4] = ts - sweep_ts
                # points_sweep = points.new_point(points_sweep)
                sweep_points_list.append(points_sweep)

        points = np.concatenate(sweep_points_list, axis=0)
        if self.filter_by_angle:
            points = self.filter_point_by_angle(points)

        points = points[:, self.use_dim]
        results["points"] = points
        return results


class LoadMultiViewImageFromFiles:
    """
    load multi-view image from files

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Default: False.
        color_type (str): Color type of the file. Default: -1.
            - -1: cv2.IMREAD_UNCHANGED
            -  0: cv2.IMREAD_GRAYSCALE
            -  1: cv2.IMREAD_COLOR
    """

    def __init__(
        self,
        to_float32=False,
        project_pts_to_img_depth=False,
        cam_depth_range=[4.0, 45.0, 1.0],
        constant_std=0.5,
        imread_flag=-1,
    ):
        self.to_float32 = to_float32
        self.project_pts_to_img_depth = project_pts_to_img_depth
        self.cam_depth_range = cam_depth_range
        self.constant_std = constant_std
        self.imread_flag = imread_flag

    def __call__(self, sample):
        """
        Call function to load multi-view image from files.
        """
        filename = sample["img_filename"]

        img = np.stack(
            [cv2.imread(name, self.imread_flag) for name in filename], axis=-1
        )
        if self.to_float32:
            img = img.astype(np.float32)
        sample["filename"] = filename

        sample["img"] = [img[..., i] for i in range(img.shape[-1])]
        sample["img_shape"] = img.shape
        sample["ori_shape"] = img.shape

        sample["pad_shape"] = img.shape
        # sample['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]

        sample["img_norm_cfg"] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False,
        )
        sample["img_fields"] = ["img"]
        return sample


class ResizeImage:
    """Resize images & bbox & mask."""

    def __init__(
        self,
        img_scale=None,
        multiscale_mode="range",
        ratio_range=None,
        keep_ratio=True,
        bbox_clip_border=True,
        backend="cv2",
        override=False,
    ):
        if img_scale is None:
            self.img_scale = None
        else:
            if isinstance(img_scale, list):
                self.img_scale = img_scale
            else:
                self.img_scale = [img_scale]

        if ratio_range is not None:
            # mode 1: given a scale and a range of image ratio
            assert len(self.img_scale) == 1
        else:
            # mode 2: given multiple scales or a range of scales
            assert multiscale_mode in ["value", "range"]

        self.backend = backend
        self.multiscale_mode = multiscale_mode
        self.ratio_range = ratio_range
        self.keep_ratio = keep_ratio
        # TODO: refactor the override option in Resize
        self.override = override
        self.bbox_clip_border = bbox_clip_border

    @staticmethod
    def random_select(img_scales):
        """Randomly select an img_scale from given candidates."""
        scale_idx = np.random.randint(len(img_scales))
        img_scale = img_scales[scale_idx]
        return img_scale, scale_idx

    @staticmethod
    def random_sample(img_scales):
        """Randomly sample an img_scale when ``multiscale_mode=='range'``."""
        img_scale_long = [max(s) for s in img_scales]
        img_scale_short = [min(s) for s in img_scales]
        long_edge = np.random.randint(min(img_scale_long), max(img_scale_long) + 1)
        short_edge = np.random.randint(min(img_scale_short), max(img_scale_short) + 1)
        img_scale = (long_edge, short_edge)
        return img_scale, None

    @staticmethod
    def random_sample_ratio(img_scale, ratio_range):
        """Randomly sample an img_scale when ``ratio_range`` is specified."""

        assert isinstance(img_scale, list) and len(img_scale) == 2
        min_ratio, max_ratio = ratio_range
        assert min_ratio <= max_ratio
        ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
        scale = int(img_scale[0] * ratio), int(img_scale[1] * ratio)
        return scale, None

    def _random_scale(self, results):
        """Randomly sample an img_scale according to ``ratio_range`` and
        ``multiscale_mode``.
        """

        if self.ratio_range is not None:
            scale, scale_idx = self.random_sample_ratio(
                self.img_scale[0], self.ratio_range
            )
        elif len(self.img_scale) == 1:
            scale, scale_idx = self.img_scale[0], 0
        elif self.multiscale_mode == "range":
            scale, scale_idx = self.random_sample(self.img_scale)
        elif self.multiscale_mode == "value":
            scale, scale_idx = self.random_select(self.img_scale)
        else:
            raise NotImplementedError

        results["scale"] = scale
        results["scale_idx"] = scale_idx

    def _resize_img(self, results):
        """Resize images with ``results['scale']``."""
        for key in results.get("img_fields", ["img"]):
            for idx in range(len(results["img"])):
                if self.keep_ratio:
                    img, scale_factor = self.imrescale(
                        results[key][idx],
                        results["scale"],
                        interpolation="bilinear" if key == "img" else "nearest",
                        return_scale=True,
                        backend=self.backend,
                    )
                    new_h, new_w = img.shape[:2]
                    h, w = results[key][idx].shape[:2]
                    w_scale = new_w / w
                    h_scale = new_h / h
                else:
                    raise NotImplementedError
                results[key][idx] = img

            scale_factor = np.array(
                [w_scale, h_scale, w_scale, h_scale], dtype=np.float32
            )
            results["img_shape"] = img.shape
            # in case that there is no padding
            results["pad_shape"] = img.shape
            results["scale_factor"] = scale_factor
            results["keep_ratio"] = self.keep_ratio

    def rescale_size(self, old_size, scale, return_scale=False):
        """Calculate the new size to be rescaled to."""
        w, h = old_size
        if isinstance(scale, (float, int)):
            if scale <= 0:
                raise ValueError(f"Invalid scale {scale}, must be positive.")
            scale_factor = scale
        elif isinstance(scale, list):
            max_long_edge = max(scale)
            max_short_edge = min(scale)
            scale_factor = min(max_long_edge / max(h, w), max_short_edge / min(h, w))
        else:
            raise TypeError(
                f"Scale must be a number or list of int, but got {type(scale)}"
            )

        def _scale_size(size, scale):
            if isinstance(scale, (float, int)):
                scale = (scale, scale)
            w, h = size
            return int(w * float(scale[0]) + 0.5), int(h * float(scale[1]) + 0.5)

        new_size = _scale_size((w, h), scale_factor)

        if return_scale:
            return new_size, scale_factor
        else:
            return new_size

    def imrescale(
        self, img, scale, return_scale=False, interpolation="bilinear", backend=None
    ):
        """Resize image while keeping the aspect ratio."""
        h, w = img.shape[:2]
        new_size, scale_factor = self.rescale_size((w, h), scale, return_scale=True)
        rescaled_img = self.imresize(
            img, new_size, interpolation=interpolation, backend=backend
        )
        if return_scale:
            return rescaled_img, scale_factor
        else:
            return rescaled_img

    def imresize(
        self,
        img,
        size,
        return_scale=False,
        interpolation="bilinear",
        out=None,
        backend=None,
    ):
        """Resize image to a given size."""
        h, w = img.shape[:2]
        if backend not in ["cv2", "pillow"]:
            raise ValueError(
                f"backend: {backend} is not supported for resize."
                f"Supported backends are 'cv2', 'pillow'"
            )

        if backend == "pillow":
            raise NotImplementedError
        else:
            resized_img = cv2.resize(
                img, size, dst=out, interpolation=cv2_interp_codes[interpolation]
            )
        if not return_scale:
            return resized_img
        else:
            w_scale = size[0] / w
            h_scale = size[1] / h
            return resized_img, w_scale, h_scale

    def _resize_bboxes(self, results):
        """Resize bounding boxes with ``results['scale_factor']``."""
        for key in results.get("bbox_fields", []):
            bboxes = results[key] * results["scale_factor"]
            if self.bbox_clip_border:
                img_shape = results["img_shape"]
                bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
                bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])
            results[key] = bboxes

    def _resize_masks(self, results):
        """Resize masks with ``results['scale']``"""
        raise NotImplementedError

    def _resize_seg(self, results):
        """Resize semantic segmentation map with ``results['scale']``."""
        raise NotImplementedError

    def __call__(self, results):
        """Call function to resize images, bounding boxes, masks, semantic
        segmentation map.
        """
        if "scale" not in results:
            if "scale_factor" in results:
                img_shape = results["img"][0].shape[:2]
                scale_factor = results["scale_factor"]
                assert isinstance(scale_factor, float)
                results["scale"] = list(
                    [int(x * scale_factor) for x in img_shape][::-1]
                )
            else:
                self._random_scale(results)
        else:
            if not self.override:
                assert (
                    "scale_factor" not in results
                ), "scale and scale_factor cannot be both set."
            else:
                results.pop("scale")
                if "scale_factor" in results:
                    results.pop("scale_factor")
                self._random_scale(results)

        self._resize_img(results)
        self._resize_bboxes(results)
        return results


class NormalizeImage:
    """Normalize the image."""

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def _imnormalize(self, img, mean, std, to_rgb=True):
        img = img.copy().astype(np.float32)
        mean = np.float64(mean.reshape(1, -1))
        stdinv = 1 / np.float64(std.reshape(1, -1))
        if to_rgb:
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)  # inplace
        cv2.subtract(img, mean, img)  # inplace
        cv2.multiply(img, stdinv, img)  # inplace
        return img

    def __call__(self, results):
        """Call function to normalize images."""
        for key in results.get("img_fields", ["img"]):
            if key == "img_depth":
                continue
            for idx in range(len(results["img"])):
                results[key][idx] = self._imnormalize(
                    results[key][idx], self.mean, self.std, self.to_rgb
                )
        results["img_norm_cfg"] = dict(mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results


class PadImage(object):
    """Pad the image & mask."""

    def __init__(self, size=None, size_divisor=None, pad_val=0):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        # only one of size and size_divisor should be valid
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    def impad(
        self, img, *, shape=None, padding=None, pad_val=0, padding_mode="constant"
    ):
        """Pad the given image to a certain shape or pad on all sides with
        specified padding mode and padding value.
        """

        assert (shape is not None) ^ (padding is not None)
        if shape is not None:
            padding = [0, 0, shape[1] - img.shape[1], shape[0] - img.shape[0]]

        # check pad_val
        if isinstance(pad_val, list):
            assert len(pad_val) == img.shape[-1]
        elif not isinstance(pad_val, numbers.Number):
            raise TypeError(
                "pad_val must be a int or a list. " f"But received {type(pad_val)}"
            )

        # check padding
        if isinstance(padding, list) and len(padding) in [2, 4]:
            if len(padding) == 2:
                padding = [padding[0], padding[1], padding[0], padding[1]]
        elif isinstance(padding, numbers.Number):
            padding = [padding, padding, padding, padding]
        else:
            raise ValueError(
                "Padding must be a int or a 2, or 4 element list."
                f"But received {padding}"
            )

        # check padding mode
        assert padding_mode in ["constant", "edge", "reflect", "symmetric"]

        border_type = {
            "constant": cv2.BORDER_CONSTANT,
            "edge": cv2.BORDER_REPLICATE,
            "reflect": cv2.BORDER_REFLECT_101,
            "symmetric": cv2.BORDER_REFLECT,
        }
        img = cv2.copyMakeBorder(
            img,
            padding[1],
            padding[3],
            padding[0],
            padding[2],
            border_type[padding_mode],
            value=pad_val,
        )

        return img

    def impad_to_multiple(self, img, divisor, pad_val=0):
        """Pad an image to ensure each edge to be multiple to some number."""
        pad_h = int(np.ceil(img.shape[0] / divisor)) * divisor
        pad_w = int(np.ceil(img.shape[1] / divisor)) * divisor
        return self.impad(img, shape=(pad_h, pad_w), pad_val=pad_val)

    def _pad_img(self, results):
        """Pad images according to ``self.size``."""
        for key in results.get("img_fields", ["img"]):
            if self.size is not None:
                padded_img = self.impad(
                    results[key], shape=self.size, pad_val=self.pad_val
                )
            elif self.size_divisor is not None:
                for idx in range(len(results[key])):
                    padded_img = self.impad_to_multiple(
                        results[key][idx], self.size_divisor, pad_val=self.pad_val
                    )
                    results[key][idx] = padded_img
        results["pad_shape"] = padded_img.shape
        results["pad_fixed_size"] = self.size
        results["pad_size_divisor"] = self.size_divisor

    def _pad_masks(self, results):
        """Pad masks according to ``results['pad_shape']``."""
        raise NotImplementedError

    def _pad_seg(self, results):
        """Pad semantic segmentation map according to
        ``results['pad_shape']``."""
        raise NotImplementedError

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps."""
        self._pad_img(results)
        return results


class SampleFilterByKey:
    """Collect data from the loader relevant to the specific task."""

    def __init__(
        self,
        keys,
        meta_keys=(
            "filename",
            "ori_shape",
            "img_shape",
            "lidar2img",
            "depth2img",
            "cam2img",
            "pad_shape",
            "scale_factor",
            "flip",
            "pcd_horizontal_flip",
            "pcd_vertical_flip",
            "box_type_3d",
            "img_norm_cfg",
            "pcd_trans",
            "sample_idx",
            "pcd_scale_factor",
            "pcd_rotation",
            "pts_filename",
            "transformation_3d_flow",
        ),
    ):
        self.keys = keys
        self.meta_keys = meta_keys

    def __call__(self, sample):
        """Call function to filter sample by keys. The keys in ``meta_keys``"""
        filtered_sample = Sample(path=sample.path, modality=sample.modality)
        filtered_sample.meta.id = sample.meta.id
        img_metas = {}

        for key in self.meta_keys:
            if key in sample:
                img_metas[key] = sample[key]

        filtered_sample["img_metas"] = img_metas
        for key in self.keys:
            filtered_sample[key] = sample[key]

        return filtered_sample


class GetInferInput:
    """Collect infer input data from transformed sample"""

    def collate_fn(self, batch):
        sample = batch[0]
        collated_batch = {}
        collated_fields = [
            "img",
            "points",
            "img_metas",
            "gt_bboxes_3d",
            "gt_labels_3d",
            "modality",
            "meta",
            "idx",
            "img_depth",
        ]
        for k in list(sample.keys()):
            if k not in collated_fields:
                continue
            if k == "img":
                collated_batch[k] = np.stack([elem[k] for elem in batch], axis=0)
            elif k == "img_depth":
                collated_batch[k] = np.stack(
                    [np.stack(elem[k], axis=0) for elem in batch], axis=0
                )
            else:
                collated_batch[k] = [elem[k] for elem in batch]
        return collated_batch

    def __call__(self, sample):
        """Call function to filter sample by keys. The keys in ``meta_keys``"""
        if sample.modality == "multimodal" or sample.modality == "multiview":
            if "img" in sample.keys():
                sample.img = np.stack(
                    [img.transpose(2, 0, 1) for img in sample.img], axis=0
                )

        sample = self.collate_fn([sample])
        infer_input = []
        points = sample.get("points", None)[0]
        infer_input.append(points.astype(np.float32))
        img = sample.get("img", None)[0]
        infer_input.append(img.astype(np.float32))
        lidar2img = np.stack(sample["img_metas"][0]["lidar2img"]).astype(np.float32)
        infer_input.append(lidar2img)
        sample_id = sample["img_metas"][0]["sample_idx"]

        return infer_input, sample_id