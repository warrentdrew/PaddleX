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

from ...common.result import BaseResult, StrMixin, JsonMixin


class BEV3DDetResult(BaseResult, StrMixin, JsonMixin):
    """Base class for computer vision results."""

    def __init__(self, data: dict) -> None:
        """
        Initialize the BaseCVResult.

        Args:
            data (dict): The initial data.

        Raises:
            AssertionError: If the required key (`BaseCVResult.INPUT_IMG_KEY`) are not found in the data.
        """

        super().__init__(data)
        StrMixin.__init__(self)
        JsonMixin.__init__(self)
