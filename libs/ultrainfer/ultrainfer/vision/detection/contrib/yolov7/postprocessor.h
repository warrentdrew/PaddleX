// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#include "ultrainfer/vision/common/processors/transform.h"
#include "ultrainfer/vision/common/result.h"

namespace ultrainfer {
namespace vision {

namespace detection {
/*! @brief Postprocessor object for YOLOv7 serials model.
 */
class ULTRAINFER_DECL YOLOv7Postprocessor {
public:
  /** \brief Create a postprocessor instance for YOLOv7 serials model
   */
  YOLOv7Postprocessor();

  /** \brief Process the result of runtime and fill to DetectionResult structure
   *
   * \param[in] tensors The inference result from runtime
   * \param[in] result The output result of detection
   * \param[in] ims_info The shape info list, record input_shape and
   * output_shape \return true if the postprocess successed, otherwise false
   */
  bool
  Run(const std::vector<FDTensor> &tensors,
      std::vector<DetectionResult> *results,
      const std::vector<std::map<std::string, std::array<float, 2>>> &ims_info);

  /// Set conf_threshold, default 0.25
  void SetConfThreshold(const float &conf_threshold) {
    conf_threshold_ = conf_threshold;
  }

  /// Get conf_threshold, default 0.25
  float GetConfThreshold() const { return conf_threshold_; }

  /// Set nms_threshold, default 0.5
  void SetNMSThreshold(const float &nms_threshold) {
    nms_threshold_ = nms_threshold;
  }

  /// Get nms_threshold, default 0.5
  float GetNMSThreshold() const { return nms_threshold_; }

protected:
  float conf_threshold_;
  float nms_threshold_;
  float max_wh_;
};

} // namespace detection
} // namespace vision
} // namespace ultrainfer