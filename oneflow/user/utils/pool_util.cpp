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
#include "oneflow/user/utils/pool_util.h"
#include "oneflow/core/operator/operator_util.h"

namespace oneflow {

Params3D::Params3D(const int32_t dim, const ShapeView& x_shape, const std::string& data_format,
                   const std::string& padding, const std::vector<int32_t>& padding_before,
                   const std::vector<int32_t>& padding_after, const std::vector<int32_t>& pool_size,
                   const std::vector<int32_t>& strides, const bool ceil_mode)
    : dim_(dim),
      pool_size_3d_(Get3DVec(pool_size, dim)),
      strides_3d_(Get3DVec(strides, dim)),
      padding_before_3d_(Get3DVec<Get3DVecType::kPad>(padding_before, dim)),
      padding_after_3d_(Get3DVec<Get3DVecType::kPad>(padding_after, dim)),
      data_format_(data_format),
      padding_(padding),
      ceil_mode_(ceil_mode) {
  x_3d_ = {GetInDim(x_shape, data_format, 0, dim), GetInDim(x_shape, data_format, 1, dim),
           GetInDim(x_shape, data_format, 2, dim)};
  Get3DOutputSize(x_3d_, pool_size_3d_, strides_3d_, padding_, ceil_mode_, nullptr, &y_3d_,
                  &padding_before_3d_, &padding_after_3d_);
  if (data_format == "channels_first") {
    channel_num_ = x_shape.At(1);
  } else {
    CHECK_EQ(data_format_, "channels_last")
        << "data_format must be 'channels_first' or 'channels_last'";
    channel_num_ = x_shape.At(x_shape.NumAxes() - 1);
  }
  batch_num_ = x_shape.At(0);
}

void Params3D::Reset(const ShapeView& x_shape) {
  x_3d_ = {GetInDim(x_shape, data_format_, 0, dim_), GetInDim(x_shape, data_format_, 1, dim_),
           GetInDim(x_shape, data_format_, 2, dim_)};
  Get3DOutputSize(x_3d_, pool_size_3d_, strides_3d_, padding_, ceil_mode_, nullptr, &y_3d_,
                  &padding_before_3d_, &padding_after_3d_);
}

Shape Params3D::GetYShape() const {
  DimVector y_dim_vec;
  if (dim_ == 1) {
    y_dim_vec = {y_3d_.at(2)};
  } else if (dim_ == 2) {
    y_dim_vec = {y_3d_.at(1), y_3d_.at(2)};
  } else if (dim_ == 3) {
    y_dim_vec = {y_3d_.at(0), y_3d_.at(1), y_3d_.at(2)};
  } else {
    UNIMPLEMENTED();
  }
  if (data_format_ == "channels_first") {
    y_dim_vec.insert(y_dim_vec.begin(), channel_num_);
  } else {
    CHECK_EQ(data_format_, "channels_last")
        << "data_format must be 'channels_first' or 'channels_last'";
    y_dim_vec.insert(y_dim_vec.end(), channel_num_);
  }
  y_dim_vec.insert(y_dim_vec.begin(), batch_num_);
  return Shape(y_dim_vec);
}

Shape Params3D::GetXShape5D() const {
  return Shape({batch_num_, channel_num_, x_3d_.at(0), x_3d_.at(1), x_3d_.at(2)});
}

Shape Params3D::GetYShape5D() const {
  return Shape({batch_num_, channel_num_, y_3d_.at(0), y_3d_.at(1), y_3d_.at(2)});
}

}  // namespace oneflow
