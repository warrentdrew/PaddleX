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

import numpy as np
import cv2

from ...common.result import BaseDet3DResult


class BEV3DDetResult(BaseDet3DResult):

    def __init__(self, data):
        super().__init__(data)

    def _to_img(self):
        image = self._input_img
        bboxes = self["boxes_3d"]
        labels = self["labels_3d"]
        scores = self["scores_3d"]
