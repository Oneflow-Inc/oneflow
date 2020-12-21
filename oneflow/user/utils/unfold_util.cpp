/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include "oneflow/user/utils/unfold_util.h"
#include "oneflow/core/operator/operator_util.h"

namespace oneflow {

namespace user_op {

namespace {

std::vector<int32_t> Get3DVec(const std::vector<int32_t>& original_vec, int32_t NDims) {
  std::vector<int32_t> vec;
  FOR_RANGE(uint8_t, dim, 0, 3) {
    int64_t index = static_cast<int64_t>(dim) - (3 - NDims);
    if (index < 0) {
      vec.push_back(1);
    } else {
      vec.push_back(original_vec.at(index));
    }
  }
  return vec;
}

std::vector<int32_t> Get3DPadVec(const std::vector<int32_t>& original_vec, int32_t NDims) {
  std::vector<int32_t> vec;
  FOR_RANGE(uint8_t, dim, 0, 3) {
    int64_t index = static_cast<int64_t>(dim) - (3 - NDims);
    if (index < 0) {
      vec.push_back(0);
    } else {
      vec.push_back(original_vec.at(index));
    }
  }
  return vec;
}

}  // namespace

ParamsUnfold3D::ParamsUnfold3D(const int32_t dim, const ShapeView& x_shape,
                               const std::string& data_format, const std::string& padding,
                               const std::vector<int32_t>& padding_before,
                               const std::vector<int32_t>& padding_after,
                               const std::vector<int32_t>& kernel_size,
                               const std::vector<int32_t>& strides,
                               const std::vector<int32_t>& dilation_rate, const bool ceil_mode)
    : dim_(dim),
      kernel_size_3d_(Get3DVec(kernel_size, dim)),
      strides_3d_(Get3DVec(strides, dim)),
      dilation_rate_3d_(Get3DVec(dilation_rate, dim)),
      padding_before_3d_(Get3DPadVec(padding_before, dim)),
      padding_after_3d_(Get3DPadVec(padding_after, dim)),
      data_format_(data_format),
      padding_(padding),
      ceil_mode_(ceil_mode) {
  x_3d_ = {GetInDim(x_shape, data_format, 0, dim), GetInDim(x_shape, data_format, 1, dim),
           GetInDim(x_shape, data_format, 2, dim)};
  Get3DOutputSize(x_3d_, kernel_size_3d_, strides_3d_, padding_, ceil_mode_, &dilation_rate_3d_,
                  &y_3d_, &padding_before_3d_, &padding_after_3d_);
  if (data_format == "channels_first") {
    channel_num_ = x_shape.At(1);
  } else {
    CHECK_EQ(data_format_, "channels_last")
        << "data_format must be 'channels_first' or 'channels_last'";
    channel_num_ = x_shape.At(x_shape.NumAxes() - 1);
  }
  batch_num_ = x_shape.At(0);
}

Shape ParamsUnfold3D::GetYShape() const {
  if (dim_ < 1 || dim_ > 3) { UNIMPLEMENTED(); }
  DimVector y_dim_vec{batch_num_, 0, 0};
  y_dim_vec.at(1) =
      channel_num_ * kernel_size_3d_.at(0) * kernel_size_3d_.at(1) * kernel_size_3d_.at(2);
  y_dim_vec.at(2) = y_3d_.at(0) * y_3d_.at(1) * y_3d_.at(2);
  return Shape(y_dim_vec);
}

Shape ParamsUnfold3D::GetXShape5D() const {
  return Shape({batch_num_, channel_num_, x_3d_.at(0), x_3d_.at(1), x_3d_.at(2)});
}

Shape ParamsUnfold3D::GetYShape5D() const {
  return Shape({batch_num_, channel_num_, y_3d_.at(0), y_3d_.at(1), y_3d_.at(2)});
}

}  // namespace user_op

}  // namespace oneflow
