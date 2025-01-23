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

from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .....utils import logging
from ...single_model_pipeline import MultiLableImageClas
from .. import utils as serving_utils
from ..app import AppConfig, create_app
from ..models import NoResultResponse, ResultResponse


class InferenceParams(BaseModel):
    threshold: Optional[float] = None


class InferRequest(BaseModel):
    image: str
    inferenceParams: Optional[InferenceParams] = None


class Category(BaseModel):
    id: int
    name: str
    score: float


class InferResult(BaseModel):
    categories: List[Category]
    image: str


def create_pipeline_app(
    pipeline: MultiLableImageClas, app_config: AppConfig
) -> FastAPI:
    app, ctx = create_app(
        pipeline=pipeline, app_config=app_config, app_aiohttp_session=True
    )

    @app.post(
        "/multilabel-image-classification",
        operation_id="infer",
        responses={422: {"model": NoResultResponse}},
        response_model_exclude_none=True,
    )
    async def _infer(request: InferRequest) -> ResultResponse[InferResult]:
        pipeline = ctx.pipeline
        aiohttp_session = ctx.aiohttp_session

        if request.inferenceParams:
            threshold = request.inferenceParams.threshold
            if threshold is not None:
                raise HTTPException(
                    status_code=422,
                    detail="`threshold` is currently not supported.",
                )

        try:
            file_bytes = await serving_utils.get_raw_bytes(
                request.image, aiohttp_session
            )
            image = serving_utils.image_bytes_to_array(file_bytes)

            result = (await pipeline.infer(image))[0]

            if "label_names" in result:
                cat_names = result["label_names"]
            else:
                cat_names = [str(id_) for id_ in result["class_ids"]]
            categories: List[Category] = []
            for id_, name, score in zip(
                result["class_ids"], cat_names, result["scores"]
            ):
                categories.append(Category(id=id_, name=name, score=score))
            output_image_base64 = serving_utils.base64_encode(
                serving_utils.image_to_bytes(result.img)
            )

            return ResultResponse[InferResult](
                logId=serving_utils.generate_log_id(),
                result=InferResult(categories=categories, image=output_image_base64),
            )

        except Exception:
            logging.exception("Unexpected exception")
            raise HTTPException(status_code=500, detail="Internal server error")

    return app
