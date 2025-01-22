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

import os
import ast
from pathlib import Path
import numpy as np

from ....utils import logging
from ....utils.download import download
from ....utils.cache import CACHE_DIR
from .base_batch_sampler import BaseBatchSampler


class VideoBatchSampler(BaseBatchSampler):

    SUFFIX = ["mp4", "avi", "mkv", "webm"]

    # XXX: auto download for url
    def _download_from_url(self, in_path):
        file_name = Path(in_path).name
        save_path = Path(CACHE_DIR) / "predict_input" / file_name
        download(in_path, save_path, overwrite=True)
        return save_path.as_posix()

    def _get_files_list(self, fp):
        file_list = []
        if fp is None or not os.path.exists(fp):
            raise Exception(f"Not found any video file in path: {fp}")
        if os.path.isfile(fp) and fp.split(".")[-1] in self.SUFFIX:
            file_list.append(fp)
        elif os.path.isdir(fp):
            for root, dirs, files in os.walk(fp):
                for single_file in files:
                    if single_file.split(".")[-1] in self.SUFFIX:
                        file_list.append(os.path.join(root, single_file))
        if len(file_list) == 0:
            raise Exception("Not found any file in {}".format(fp))
        file_list = sorted(file_list)
        return file_list

    def sample(self, inputs):
        if not isinstance(inputs, list):
            inputs = [inputs]

        batch = []
        for input in inputs:
            if isinstance(input, str):
                file_path = (
                    self._download_from_url(input)
                    if input.startswith("http")
                    else input
                )
                file_list = self._get_files_list(file_path)
                for file_path in file_list:
                    batch.append(file_path)
                    if len(batch) == self.batch_size:
                        yield batch
                        batch = []
            else:
                logging.warning(
                    f"Not supported input data type! Only `str` are supported! So has been ignored: {input}."
                )
        if len(batch) > 0:
            yield batch

    def _rand_batch(self, data_size):
        def parse_size(s):
            res = ast.literal_eval(s)
            if isinstance(res, int):
                return (res, res)
            else:
                assert isinstance(res, (tuple, list))
                assert len(res) == 2
                assert all(isinstance(item, int) for item in res)
                return res

        size = parse_size(data_size)
        rand_batch = [
            np.random.randint(0, 256, (*size, 3), dtype=np.uint8)
            for _ in range(self.batch_size)
        ]
        return rand_batch
