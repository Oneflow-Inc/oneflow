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
#include "oneflow/user/kernels/max_pool_kernel_util.h"

namespace oneflow {

void GetWindowedOutputShape(int64_t input_size, int32_t filter_size, int32_t stride,
                            int32_t padding, bool ceil_mode, int32_t dilation_rate,
                            int64_t* output_ptr) {
  int64_t output_size = (input_size + 2 * padding - dilation_rate * (filter_size - 1) - 1 + stride
                         + (ceil_mode ? stride - 1 : 0))
                        / stride;

  if (ceil_mode) {
    // ensure that the last pool starts inside the image
    // needed to avoid problems in ceil mode
    if ((output_size - 1) * stride >= input_size + padding) { --output_size; }
  }
  *output_ptr = output_size;
}

void Get3DOutputShape(const DimVector& in, const std::vector<int32_t>& pool_size,
                      const std::vector<int32_t>& strides, const std::vector<int32_t>& padding,
                      const bool ceil_mode, std::vector<int32_t> dilation_rate, DimVector* out) {
  out->clear();
  out->resize(3);
  FOR_RANGE(size_t, i, 0, 3) {
    int64_t* out_ptr = &(*out).at(i);
    GetWindowedOutputShape(in.at(i), pool_size.at(i), strides.at(i), padding.at(i), ceil_mode,
                           dilation_rate.at(i), out_ptr);
  }
}

MaxPoolParams3D::MaxPoolParams3D(const int32_t dim, const ShapeView& x_shape,
                                 const std::string& data_format,
                                 const std::vector<int32_t>& padding,
                                 const std::vector<int32_t>& kernel_size,
                                 const std::vector<int32_t>& stride,
                                 const std::vector<int32_t>& dilation, const bool return_indices,
                                 const bool ceil_mode)
    : dim_(dim),
      data_format_(data_format),
      padding_(Get3DVec<Get3DVecType::kPad>(padding, dim)),
      pool_size_3d_(Get3DVec(kernel_size, dim)),
      stride_3d_(Get3DVec(stride, dim)),
      dilation_3d_(Get3DVec(dilation, dim)),
      return_indices_(return_indices),
      ceil_mode_(ceil_mode) {
  x_3d_ = {GetInDim(x_shape, data_format, 0, dim), GetInDim(x_shape, data_format, 1, dim),
           GetInDim(x_shape, data_format, 2, dim)};
  Get3DOutputShape(x_3d_, pool_size_3d_, stride_3d_, padding_, ceil_mode_, dilation_3d_, &y_3d_);
  if (data_format == "channels_first") {
    channel_num_ = x_shape.At(1);
  } else {
    CHECK_EQ(data_format_, "channels_last")
        << "data_format must be 'channels_first' or 'channels_last'";
    channel_num_ = x_shape.At(x_shape.NumAxes() - 1);
  }
  batch_num_ = x_shape.At(0);
}

void MaxPoolParams3D::Reset(const ShapeView& x_shape) {
  x_3d_ = {GetInDim(x_shape, data_format_, 0, dim_), GetInDim(x_shape, data_format_, 1, dim_),
           GetInDim(x_shape, data_format_, 2, dim_)};
  Get3DOutputShape(x_3d_, pool_size_3d_, stride_3d_, padding_, ceil_mode_, dilation_3d_, &y_3d_);
}

Shape MaxPoolParams3D::GetYShape() const {
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

Shape MaxPoolParams3D::GetXShape5D() const {
  return Shape({batch_num_, channel_num_, x_3d_.at(0), x_3d_.at(1), x_3d_.at(2)});
}

Shape MaxPoolParams3D::GetYShape5D() const {
  return Shape({batch_num_, channel_num_, y_3d_.at(0), y_3d_.at(1), y_3d_.at(2)});
}

}  // namespace oneflow
