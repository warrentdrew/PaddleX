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

from typing import Any, Dict, Optional, Union, List
import numpy as np

from ...utils.pp_option import PaddlePredictorOption
from ..base import BasePipeline
from ...models_new.multilingual_speech_recognition.result import WhisperResult


class MultilingualSpeechRecognitionPipeline(BasePipeline):
    """Multilingual Speech Recognition Pipeline"""

    entities = "multilingual_speech_recognition"

    def __init__(
        self,
        config: Dict,
        device: str = None,
        pp_option: PaddlePredictorOption = None,
        use_hpip: bool = False,
    ) -> None:
        """
        Initializes the class with given configurations and options.

        Args:
            config (Dict): Configuration dictionary containing model and other parameters.
            device (str): The device to run the prediction on. Default is None.
            pp_option (PaddlePredictorOption): Options for PaddlePaddle predictor. Default is None.
            use_hpip (bool): Whether to use high-performance inference (hpip) for prediction. Defaults to False.
        """
        super().__init__(device=device, pp_option=pp_option, use_hpip=use_hpip)

        multilingual_speech_recognition_model_config = config["SubModules"][
            "MultilingualSpeechRecognition"
        ]
        self.multilingual_speech_recognition_model = self.create_model(
            multilingual_speech_recognition_model_config
        )
        # only support batch size 1
        batch_size = multilingual_speech_recognition_model_config["batch_size"]

    def predict(
        self, input: Union[str, List[str], np.ndarray, List[np.ndarray]], **kwargs
    ) -> WhisperResult:
        """Predicts speech recognition results for the given input.

        Args:
            input (Union[str, list[str], np.ndarray, list[np.ndarray]]): The input audio or path.
            **kwargs: Additional keyword arguments that can be passed to the function.

        Returns:
            WhisperResult: The predicted whisper results, support str and json output.
        """
        yield from self.multilingual_speech_recognition_model(input)
