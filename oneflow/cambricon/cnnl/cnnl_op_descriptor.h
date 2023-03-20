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
#ifndef ONEFLOW_CAMBRICON_CNNL_CNNL_OP_DESCRIPTOR_H_
#define ONEFLOW_CAMBRICON_CNNL_CNNL_OP_DESCRIPTOR_H_

#include "oneflow/cambricon/cnnl/cnnl_common_descriptor.h"

// Modified from Cambricon catch for PyTorch.
// https://github.com/Cambricon/catch/blob/main/torch_mlu/csrc/aten/cnnl/cnnlOpDescriptors.h

namespace oneflow {

class CnnlPoolingDescriptor : public CnnlDescriptor<cnnlPoolingStruct, &cnnlCreatePoolingDescriptor,
                                                    &cnnlDestroyPoolingDescriptor> {
 public:
  CnnlPoolingDescriptor() = default;

  void set(cnnlPoolingMode_t mode, int kernel_h, int kernel_w, int stride_h, int stride_w,
           int pad_u, int pad_d, int pad_l, int pad_r, bool ceil_mode);

  // NdPooling
  void set(cnnlPoolingMode_t mode, int64_t dims, const int kernel_size[], const int stride[],
           const int padding[]);
  void set(cnnlPoolingMode_t mode, int64_t dims, const int kernel_size[], const int stride[],
           const int padding[], const int dilation[], bool ceil_mode);
};

class CnnlTransposeDescriptor
    : public CnnlDescriptor<cnnlTransposeStruct, &cnnlCreateTransposeDescriptor,
                            &cnnlDestroyTransposeDescriptor> {
 public:
  CnnlTransposeDescriptor() {}

  void set(const int p_dims, const int permute[]);
};

class CnnlReduceDescriptor : public CnnlDescriptor<cnnlReduceStruct, &cnnlCreateReduceDescriptor,
                                                   &cnnlDestroyReduceDescriptor> {
 public:
  CnnlReduceDescriptor() {}
  void set(cnnlDataType_t data_type, std::vector<int32_t> axis, cnnlReduceOp_t mode,
           cnnlReduceIndices_t is_indices, cnnlIndicesType_t indices_type);
};

class CnnlOpTensorDescriptor
    : public CnnlDescriptor<cnnlOpTensorStruct, &cnnlCreateOpTensorDescriptor,
                            &cnnlDestroyOpTensorDescriptor> {
 public:
  CnnlOpTensorDescriptor() {}

  void set(cnnlOpTensorDesc_t op_type, cnnlDataType_t op_tensor_comp_type,
           cnnlNanPropagation_t op_tensor_nan_opt);
};

class CnnlActivationDescriptor
    : public CnnlDescriptor<cnnlActivationStruct, &cnnlCreateActivationDescriptor,
                            &cnnlDestroyActivationDescriptor> {
 public:
  CnnlActivationDescriptor() {}

  void set(cnnlActivationMode_t mode, cnnlActivationPreference_t prefer,
           cnnlNanPropagation_t nanProp, float ceof, int sliced_dim = 0, float gamma = 0.f,
           float scale = 0.f, bool is_result = false, bool approximate = true);
};

class CnnlConvolutionDescriptor
    : public CnnlDescriptor<cnnlConvolutionStruct, &cnnlCreateConvolutionDescriptor,
                            &cnnlDestroyConvolutionDescriptor> {
 public:
  CnnlConvolutionDescriptor() {}

  void set(int dim, int* stride, int* padding, int* dilation, int64_t groups, cnnlDataType_t dtype);
};

class CnnlDeconvolutionDescriptor
    : public CnnlDescriptor<cnnlDeconvolutionStruct, &cnnlCreateDeconvolutionDescriptor,
                            &cnnlDestroyDeconvolutionDescriptor> {
 public:
  CnnlDeconvolutionDescriptor() {}

  void set(int dim, int* stride, int* padding, int* dilation, int64_t groups, cnnlDataType_t dtype);
};

class CnnlMatmulDescriptor
    : public CnnlDescriptor<cnnlMatMulStruct, &cnnlMatMulDescCreate, &cnnlMatMulDescDestroy> {
 public:
  CnnlMatmulDescriptor() {}
  void set_attr(cnnlMatMulDescAttribute_t attr, const void* buf, size_t size_in_bytes);
};

class CnnlBatchMatmulDescriptor
    : public CnnlDescriptor<cnnlBatchMatMulStruct, &cnnlBatchMatMulDescCreate,
                            &cnnlBatchMatMulDescDestroy> {
 public:
  CnnlBatchMatmulDescriptor() {}
  void set_attr(cnnlBatchMatMulDescAttribute_t attr, const void* buf, size_t size_in_bytes);
};

class CnnlUniqueDescriptor : public CnnlDescriptor<cnnlUniqueStruct, &cnnlCreateUniqueDescriptor,
                                                   &cnnlDestroyUniqueDescriptor> {
 public:
  CnnlUniqueDescriptor() {}

  void set(bool sorted, int dim, bool return_inverse, bool return_counts);
};

class CnnlCTCLossDescriptor : public CnnlDescriptor<cnnlCTCLossStruct, &cnnlCreateCTCLossDescriptor,
                                                    &cnnlDestroyCTCLossDescriptor> {
 public:
  CnnlCTCLossDescriptor() {}
  void set(cnnlCTCLossNormalizationMode_t norm_mode, cnnlCTCLossReduceMode_t reduce_mode,
           cnnlCTCLossZeroInfinityMode_t zero_infinity, int blank, int max_input_length,
           int max_label_length);
};

class CnnlNmsDescriptor
    : public CnnlDescriptor<cnnlNmsStruct, &cnnlCreateNmsDescriptor, &cnnlDestroyNmsDescriptor> {
 public:
  CnnlNmsDescriptor() {}
  void set(const cnnlNmsOutputMode_t mode, const float iou_threshold, const int max_output_size,
           const float confidence_threshold, const int input_layout);
};

class CnnlTrigonDescriptor : public CnnlDescriptor<cnnlTrigonStruct, &cnnlCreateTrigonDescriptor,
                                                   &cnnlDestroyTrigonDescriptor> {
 public:
  CnnlTrigonDescriptor() {}
  void set(cnnlTrigonFunctionMode_t mode);
};

}  // namespace oneflow

#endif  // ONEFLOW_CAMBRICON_CNNL_CNNL_OP_DESCRIPTOR_H_
