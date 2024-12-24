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

from typing import Any, Dict, Optional
import numpy as np

from ...common.reader import ReadImage
from ...common.batch_sampler import ImageBatchSampler
from ...utils.pp_option import PaddlePredictorOption
from ..base import BasePipeline
from ...models_new.image_classification.result import TopkResult
from ...results import TopkResult


class ImageClassificationPipeline(BasePipeline):
    """Image Classification Pipeline"""

    entities = "image_classification"

    def __init__(
        self,
        config: Dict,
        device: str = None,
        pp_option: PaddlePredictorOption = None,
        use_hpip: bool = False,
        hpi_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initializes the class with given configurations and options.

        Args:
            config (Dict): Configuration dictionary containing model and other parameters.
            device (str): The device to run the prediction on. Default is None.
            pp_option (PaddlePredictorOption): Options for PaddlePaddle predictor. Default is None.
            use_hpip (bool): Whether to use high-performance inference (hpip) for prediction. Defaults to False.
            hpi_params (Optional[Dict[str, Any]]): HPIP specific parameters. Default is None.
        """
        super().__init__(
            device=device, pp_option=pp_option, use_hpip=use_hpip, hpi_params=hpi_params
        )

        image_classification_model_config = config["SubModules"]["ImageClassification"]
        self.image_classification_model = self.create_model(
            image_classification_model_config
        )
        batch_size = image_classification_model_config["batch_size"]
        self.batch_sampler = ImageBatchSampler(batch_size=batch_size)
        self.img_reader = ReadImage(format="BGR")

    def predict(
        self, input: str | list[str] | np.ndarray | list[np.ndarray], **kwargs
    ) -> TopkResult:
        """Predicts image classification results for the given input.

        Args:
            input (str | list[str] | np.ndarray | list[np.ndarray]): The input image(s) or path(s) to the images.
            **kwargs: Additional keyword arguments that can be passed to the function.

        Returns:
            TopkResult: The predicted top k results.
        """

        for img_id, batch_data in enumerate(self.batch_sampler(input)):
            batch_imgs = self.img_reader(batch_data)
            for topk_single_result in self.image_classification_model(batch_imgs):
                yield topk_single_result