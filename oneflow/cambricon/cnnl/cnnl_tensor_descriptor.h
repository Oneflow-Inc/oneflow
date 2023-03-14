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
#ifndef ONEFLOW_CAMBRICON_CNNL_CNNL_TENSOR_DESCRIPTOR_H_
#define ONEFLOW_CAMBRICON_CNNL_CNNL_TENSOR_DESCRIPTOR_H_

#include "oneflow/cambricon/cnnl/cnnl_common_descriptor.h"
#include "oneflow/cambricon/cnnl/cnnl_types.h"
#include "oneflow/core/framework/user_op_tensor.h"

// Modified from Cambricon catch for PyTorch.
// https://github.com/Cambricon/catch/blob/main/torch_mlu/csrc/aten/cnnl/cnnlTensorDescriptors.h

namespace oneflow {

class CnnlTensorDescriptor : public CnnlDescriptor<cnnlTensorStruct, &cnnlCreateTensorDescriptor,
                                                   &cnnlDestroyTensorDescriptor> {
 public:
  // Init create Tensor descriptor
  CnnlTensorDescriptor() = default;
  // set descriptor from tensor
  void set(const user_op::Tensor* t);
  void set(const user_op::Tensor* t, cnnlTensorLayout_t layout,
           cnnlDataType_t data_type = CNNL_DTYPE_INVALID);
  void set(const user_op::Tensor* t, cnnlDataType_t dtype);
  void set_onchip_dtype(cnnlDataType_t data_type);
  void set(int position = 0, float scale = 1.0);

  void set_dim(const user_op::Tensor* t);
  void set_dim(const user_op::Tensor* t, int inputDim);

  void set_reduce(const user_op::Tensor* t);
  void set_reduce(const user_op::Tensor* t, std::vector<int64_t> keepdim);

  // for setting pooling output tensor descriptor.
  void set(const user_op::Tensor* t, bool keep_dim, std::vector<int64_t>& keepdim_sizes,
           cnnlDataType_t dtype = CNNL_DTYPE_INVALID);

  // assigned a special shape, not use tensor shape and stride info.
  void set_additional_dim(const user_op::Tensor* t, std::vector<int>& dims);

  template<typename T>
  void set(const user_op::Tensor* t, const std::vector<T>& shape_info,
           const std::vector<T>& stride_info, cnnlTensorLayout_t layout,
           cnnlDataType_t data_type = CNNL_DTYPE_INVALID) {
    OF_CHECK(shape_info.size() == stride_info.size(), "shape size need equal to stride size.");
    int t_dim = shape_info.size();
    // data_type default value is CNNL_DTYPE_INVALID in this interface,
    // and can't transmit to cnnl. so call cnnl interface will using
    // tensor dtype value when data_type value is default.
    if (data_type == CNNL_DTYPE_INVALID) { data_type = ConvertToCnnlDataType(t->data_type()); }
    if (!t_dim) {
      t_dim = 1;
      std::vector<int> dim_array(1, 1);
      OF_CNNL_CHECK(cnnlSetTensorDescriptorEx(this->mut_desc(), CNNL_LAYOUT_ARRAY, data_type, t_dim,
                                              dim_array.data(), dim_array.data()));
      return;
    }
    std::vector<int> real_shape_info(t_dim);
    std::vector<int> real_stride_info(t_dim);
    for (int i = 0; i < t_dim; ++i) {
      real_shape_info[i] = static_cast<int>(shape_info[i]);
      real_stride_info[i] = static_cast<int>(stride_info[i]);
    }
    OF_CNNL_CHECK(cnnlSetTensorDescriptorEx(this->mut_desc(), layout, data_type, t_dim,
                                            real_shape_info.data(), real_stride_info.data()));
  }
};

class CnnlSeqDataDescriptor : public CnnlDescriptor<cnnlSeqDataStruct, &cnnlCreateSeqDataDescriptor,
                                                    &cnnlDestroySeqDataDescriptor> {
 public:
  CnnlSeqDataDescriptor() {}

  void set(const user_op::Tensor* t);
  void set(const user_op::Tensor* t, cnnlSeqDataLayout_t layout);
  void set_onchip_dtype(cnnlDataType_t onchip_dtype);
};

}  // namespace oneflow

#endif  // ONEFLOW_CAMBRICON_CNNL_CNNL_TENSOR_DESCRIPTOR_H_
