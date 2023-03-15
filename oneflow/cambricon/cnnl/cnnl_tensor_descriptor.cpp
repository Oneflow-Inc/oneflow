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
#include "oneflow/cambricon/cnnl/cnnl_tensor_descriptor.h"

#include "oneflow/core/common/throw.h"

// Modified from Cambricon catch for PyTorch.
// https://github.com/Cambricon/catch/blob/main/torch_mlu/csrc/aten/cnnl/cnnlTensorDescriptors.cpp

namespace oneflow {

void CnnlTensorDescriptor::set_reduce(const user_op::Tensor* t) {
  int t_dim = t->shape_view().NumAxes();
  std::vector<int> dim_array;
  if (t_dim == 0) {
    t_dim = 1;
    dim_array.push_back(1);
  } else {
    const auto& shape = t->shape_view();
    for (int i = 0; i < t_dim; i++) { dim_array.push_back(shape.At(i)); }
  }
  auto data_type = ConvertToCnnlDataType(t->data_type());
  OF_CNNL_CHECK(cnnlSetTensorDescriptor(this->mut_desc(), CNNL_LAYOUT_NCHW, data_type, t_dim,
                                        dim_array.data()));
}
void CnnlTensorDescriptor::set_reduce(const user_op::Tensor* t, std::vector<int64_t> keepdim) {
  int t_dim = keepdim.size();
  std::vector<int> dim_array;
  if (t_dim == 0) {
    t_dim = 1;
    dim_array.push_back(1);
  } else {
    for (int i = 0; i < t_dim; i++) { dim_array.push_back(keepdim[i]); }
  }
  auto data_type = ConvertToCnnlDataType(t->data_type());
  OF_CNNL_CHECK(cnnlSetTensorDescriptor(this->mut_desc(), CNNL_LAYOUT_NCHW, data_type, t_dim,
                                        dim_array.data()));
}

void CnnlTensorDescriptor::set(const user_op::Tensor* t) {
  cnnlDataType_t data_type = ConvertToCnnlDataType(t->data_type());
  set(t, data_type);
}

void CnnlTensorDescriptor::set(const user_op::Tensor* t, cnnlDataType_t data_type) {
  int t_dim = t->shape_view().NumAxes();
  if (!t_dim) {
    t_dim = 1;
    std::vector<int> dim_array(1, 1);
    // (sg) change CNNL_LAYOUT_NHWC to CNNL_LAYOUT_ARRAY?
    OF_CNNL_CHECK(cnnlSetTensorDescriptorEx(this->mut_desc(), CNNL_LAYOUT_ARRAY, data_type, t_dim,
                                            dim_array.data(), dim_array.data()));
    return;
  }
  std::vector<int> shape_info(t_dim);
  std::vector<int> stride_info(t_dim);
  for (size_t i = 0; i < t_dim; ++i) {
    shape_info[i] = static_cast<int>(t->shape_view().At(i));
    stride_info[i] = static_cast<int>(t->stride()[i]);
  }
  OF_CNNL_CHECK(cnnlSetTensorDescriptorEx(this->mut_desc(), CNNL_LAYOUT_ARRAY, data_type, t_dim,
                                          shape_info.data(), stride_info.data()));
}

void CnnlTensorDescriptor::set(const user_op::Tensor* t, cnnlTensorLayout_t layout,
                               cnnlDataType_t data_type) {
  int t_dim = t->shape_view().NumAxes();
  if (data_type == CNNL_DTYPE_INVALID) { data_type = ConvertToCnnlDataType(t->data_type()); }
  if (!t_dim) {
    t_dim = 1;
    std::vector<int> dim_array(1, 1);
    OF_CNNL_CHECK(cnnlSetTensorDescriptorEx(this->mut_desc(), CNNL_LAYOUT_ARRAY, data_type, t_dim,
                                            dim_array.data(), dim_array.data()));
    return;
  }
  std::vector<int> shape_info(t_dim);
  std::vector<int> stride_info(t_dim);
  if (layout == CNNL_LAYOUT_NHWC || layout == CNNL_LAYOUT_NDHWC || layout == CNNL_LAYOUT_NLC) {
    for (size_t i = 0; i < t_dim; ++i) {
      shape_info[i] = static_cast<int>(t->shape_view().At(i));
      stride_info[i] = static_cast<int>(t->stride()[i]);
    }
    convertShapeAndStride(shape_info, stride_info);
  } else if (layout == CNNL_LAYOUT_HWCN) {
    // HWCN is only used by depthwise conv now, and the dim is 4
    CHECK_EQ_OR_THROW(t_dim, 4) << "depthwise convolution input's dim must be 4";
    auto convertDepthWiseConvShapeStride = [](const int64_t* vec, std::vector<int>& target_vec) {
      target_vec[0] = static_cast<int>(vec[2]);
      target_vec[1] = static_cast<int>(vec[3]);
      target_vec[2] = static_cast<int>(vec[1]);
      target_vec[3] = static_cast<int>(vec[0]);
    };
    convertDepthWiseConvShapeStride(t->shape_view().ptr(), shape_info);
    convertDepthWiseConvShapeStride(t->stride().data(), stride_info);
  } else if (layout == CNNL_LAYOUT_TNC) {
    // TNC layout is similar to ARRAY
    for (size_t i = 0; i < t_dim; ++i) {
      shape_info[i] = static_cast<int>(t->shape_view().At(i));
      stride_info[i] = static_cast<int>(t->stride()[i]);
    }
  } else {
    for (size_t i = 0; i < t_dim; ++i) {
      shape_info[i] = static_cast<int>(t->shape_view().At(i));
      stride_info[i] = static_cast<int>(t->stride()[i]);
    }
  }
  OF_CNNL_CHECK(cnnlSetTensorDescriptorEx(this->mut_desc(), layout, data_type, t_dim,
                                          shape_info.data(), stride_info.data()));
}

void CnnlTensorDescriptor::set(int position, float scale) {
  if (scale == 1.0f) {
    OF_CNNL_CHECK(cnnlSetTensorDescriptorPosition(this->mut_desc(), position));
  } else {
    OF_CNNL_CHECK(cnnlSetTensorDescriptorPositionAndScale(this->mut_desc(), position, scale));
  }
}

void CnnlTensorDescriptor::set_onchip_dtype(cnnlDataType_t onchip_dtype) {
  OF_CNNL_CHECK(cnnlSetTensorDescriptorOnchipDataType(this->mut_desc(), onchip_dtype));
}

void CnnlTensorDescriptor::set_additional_dim(const user_op::Tensor* t, std::vector<int>& dims) {
  const int dim = dims.size();
  cnnlDataType_t data_type = ConvertToCnnlDataType(t->data_type());
  std::vector<int> stride_info(dim);
  int value = 1;
  for (size_t i = dim - 1; i > 0; --i) {
    stride_info[i] = value;
    value *= dims[i];
  }
  stride_info[0] = value;
  // NCHW -> NHWC layout
  convertShapeAndStride(dims, stride_info);
  OF_CNNL_CHECK(cnnlSetTensorDescriptorEx(this->mut_desc(), CNNL_LAYOUT_NHWC, data_type, dim,
                                          dims.data(), stride_info.data()));
}

void CnnlTensorDescriptor::set_reshape(const user_op::Tensor* t, std::vector<int>& dims) {
  // TODO(WangYi): support non contiguous tensor.
  CHECK_OR_THROW(one::IsContiguous(t->shape_view(), t->stride()))
      << "set_reshape(): tensor must be contiguous";
  const int dim = dims.size();
  cnnlDataType_t data_type = ConvertToCnnlDataType(t->data_type());
  std::vector<int> stride_info(dim);
  int value = 1;
  for (size_t i = dim - 1; i > 0; --i) {
    stride_info[i] = value;
    value *= dims[i];
  }
  stride_info[0] = value;
  OF_CNNL_CHECK(cnnlSetTensorDescriptorEx(this->mut_desc(), CNNL_LAYOUT_NCHW, data_type, dim,
                                          dims.data(), stride_info.data()));
}

// Just for pooling
void CnnlTensorDescriptor::set(const user_op::Tensor* t, bool keep_dim,
                               std::vector<int64_t>& keepdim_sizes, cnnlDataType_t data_type) {
  if (data_type == CNNL_DTYPE_INVALID) { data_type = ConvertToCnnlDataType(t->data_type()); }
  int t_dim = t->shape_view().NumAxes();
  if (!keep_dim) { t_dim = keepdim_sizes.size(); }
  std::vector<int> shape_info(t_dim);
  std::vector<int> stride_info(t_dim);
  for (size_t i = 0; i < t_dim; ++i) {
    if (keep_dim) {
      shape_info[i] = static_cast<int>(t->shape_view().At(i));
      stride_info[i] = static_cast<int>(t->stride()[i]);
    } else {
      shape_info[i] = static_cast<int>(keepdim_sizes[i]);
    }
  }
  if (!keep_dim) {
    int value = 1;
    for (size_t i = t_dim - 1; i > 0; --i) {
      stride_info[i] = value;
      value *= shape_info[i];
    }
    stride_info[0] = value;
  }
  convertShapeAndStride(shape_info, stride_info);
  OF_CNNL_CHECK(cnnlSetTensorDescriptorEx(this->mut_desc(), CNNL_LAYOUT_ARRAY, data_type, t_dim,
                                          shape_info.data(), stride_info.data()));
}

void CnnlTensorDescriptor::set_dim(const user_op::Tensor* t, int inputDim) {
  cnnlDataType_t data_type = ConvertToCnnlDataType(t->data_type());
  int t_dim = t->shape_view().NumAxes();
  if (!t_dim) {
    t_dim = 1;
    std::vector<int> dim_array(1, 1);
    OF_CNNL_CHECK(cnnlSetTensorDescriptorEx(this->mut_desc(), CNNL_LAYOUT_ARRAY, data_type, t_dim,
                                            dim_array.data(), dim_array.data()));
    return;
  }
  std::vector<int> cur_size(t_dim);
  for (size_t i = 0; i < t_dim; ++i) { cur_size[i] = static_cast<int>(t->shape_view().At(i)); }
  CHECK_EQ_OR_THROW(inputDim, 4) << "inputDim need equal to 4.";
  std::vector<int> cnnl_shape_size(inputDim, 1);
  std::vector<int> cnnl_stride_size(inputDim, 1);
  for (size_t i = 0; i < inputDim; ++i) { cnnl_shape_size[i] = t_dim > i ? cur_size[i] : 1; }
  cnnl_stride_size[3] = 1;
  cnnl_stride_size[2] = cnnl_shape_size[3];
  cnnl_stride_size[1] = cnnl_stride_size[2] * cnnl_shape_size[2];
  cnnl_stride_size[0] = cnnl_stride_size[1] * cnnl_shape_size[1];
  OF_CNNL_CHECK(cnnlSetTensorDescriptorEx(this->mut_desc(), CNNL_LAYOUT_ARRAY, data_type, inputDim,
                                          cnnl_shape_size.data(), cnnl_stride_size.data()));
}

void CnnlTensorDescriptor::set_dim(const user_op::Tensor* t) {
  const int inputDim = 1;
  cnnlDataType_t data_type = ConvertToCnnlDataType(t->data_type());
  std::vector<int> cnnl_size;
  cnnl_size.push_back(t->shape_view().elem_cnt());
  std::vector<int> stride_size = {1};
  OF_CNNL_CHECK(cnnlSetTensorDescriptorEx(this->mut_desc(), CNNL_LAYOUT_ARRAY, data_type, inputDim,
                                          cnnl_size.data(), stride_size.data()));
}

void CnnlSeqDataDescriptor::set(const user_op::Tensor* t) {
  CHECK_EQ_OR_THROW(t->shape_view().NumAxes(), 3) << "input's dim must be 3.";
  auto layout = CNNL_SEQDATA_TNC;
  cnnlDataType_t dtype = ConvertToCnnlDataType(t->data_type());
  std::vector<int> dim_array(3, 1);  // TNC
  dim_array[0] = static_cast<int>(t->shape_view().At(0));
  dim_array[1] = static_cast<int>(t->shape_view().At(1));
  dim_array[2] = static_cast<int>(t->shape_view().At(2));
  auto seqLengthArraySize = 0;
  OF_CNNL_CHECK(cnnlSetSeqDataDescriptor(mut_desc(), layout, dtype, 3, dim_array.data(),
                                         seqLengthArraySize, nullptr, nullptr));
}

void CnnlSeqDataDescriptor::set(const user_op::Tensor* t, cnnlSeqDataLayout_t layout) {
  cnnlDataType_t data_type = ConvertToCnnlDataType(t->data_type());
  // t shape is NBTC
  CHECK_EQ_OR_THROW(t->shape_view().NumAxes(), 4) << "input's dim must be 4.";
  std::vector<int> dim_array(4, 1);                        // NBTC
  dim_array[0] = static_cast<int>(t->shape_view().At(0));  // N
  dim_array[1] = static_cast<int>(t->shape_view().At(1));  // B
  dim_array[2] = static_cast<int>(t->shape_view().At(2));  // T
  dim_array[3] = static_cast<int>(t->shape_view().At(3));  // C

  int seqLengthArraySize = dim_array[0] * 1;  // batch × beam

  // N is batch, B is beam, T is sequence length, C is embedding size.
  OF_CNNL_CHECK(cnnlSetSeqDataDescriptor(mut_desc(), layout, data_type, 4, dim_array.data(),
                                         seqLengthArraySize, nullptr, nullptr));
}

void CnnlSeqDataDescriptor::set_onchip_dtype(cnnlDataType_t onchip_dtype) {
  OF_CNNL_CHECK(cnnlSetSeqDataDescriptorOnchipDataType(mut_desc(), onchip_dtype));
}

}  // namespace oneflow
