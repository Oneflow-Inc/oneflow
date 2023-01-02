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
#include "oneflow/core/operator/operator_util.h"
#include "oneflow/user/utils/pool_util.h"
#include "oneflow/user/kernels/max_unpool_kernel_util.h"

namespace oneflow {
namespace {

void GetWindowedOutputShape(int64_t input_size, int32_t filter_size, int32_t stride,
                            int32_t padding, int64_t* output_ptr) {
  int64_t output_size = (input_size - 1) * stride - 2 * padding + filter_size;
  *output_ptr = output_size;
}

void Get3DOutputShape(const DimVector& in, const std::vector<int32_t>& pool_size,
                      const std::vector<int32_t>& strides, const std::vector<int32_t>& padding,
                      DimVector* out) {
  out->clear();
  out->resize(3);
  FOR_RANGE(size_t, i, 0, 3) {
    int64_t* out_ptr = &(*out).at(i);
    GetWindowedOutputShape(in.at(i), pool_size.at(i), strides.at(i), padding.at(i), out_ptr);
  }
}

}  // namespace

MaxUnpoolParams3D::MaxUnpoolParams3D(const int32_t dim, const ShapeView& x_shape,
                                     const std::vector<int32_t>& padding,
                                     const std::vector<int32_t>& kernel_size,
                                     const std::vector<int32_t>& stride)
    : dim_(dim),
      padding_(Get3DVec<Get3DVecType::kPad>(padding, dim)),
      pool_size_3d_(Get3DVec(kernel_size, dim)),
      stride_3d_(Get3DVec(stride, dim)),
      batch_num_(x_shape.At(0)),
      channel_num_(x_shape.At(1)) {
  std::string data_format = "channels_first";
  x_3d_ = {GetInDim(x_shape, data_format, 0, dim), GetInDim(x_shape, data_format, 1, dim),
           GetInDim(x_shape, data_format, 2, dim)};
  Get3DOutputShape(x_3d_, pool_size_3d_, stride_3d_, padding_, &y_3d_);
}

void MaxUnpoolParams3D::Reset(const ShapeView& x_shape) {
  std::string data_format = "channels_first";
  x_3d_ = {GetInDim(x_shape, data_format, 0, dim_), GetInDim(x_shape, data_format, 1, dim_),
           GetInDim(x_shape, data_format, 2, dim_)};
  Get3DOutputShape(x_3d_, pool_size_3d_, stride_3d_, padding_, &y_3d_);
}

int64_t MaxUnpoolParams3D::GetYStride() const { return y_3d_.at(0) * y_3d_.at(1) * y_3d_.at(2); }

Shape MaxUnpoolParams3D::GetYShape() const {
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
  y_dim_vec.insert(y_dim_vec.begin(), channel_num_);
  y_dim_vec.insert(y_dim_vec.begin(), batch_num_);
  return Shape(y_dim_vec);
}

Shape MaxUnpoolParams3D::GetXShape5D() const {
  return Shape({batch_num_, channel_num_, x_3d_.at(0), x_3d_.at(1), x_3d_.at(2)});
}

Shape MaxUnpoolParams3D::GetYShape5D() const {
  return Shape({batch_num_, channel_num_, y_3d_.at(0), y_3d_.at(1), y_3d_.at(2)});
}

}  // namespace oneflow
