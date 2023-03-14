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
#include "oneflow/cambricon/cnnl/cnnl_op_descriptor.h"

#include "oneflow/core/common/throw.h"

// Modified from Cambricon catch for PyTorch.
// https://github.com/Cambricon/catch/blob/main/torch_mlu/csrc/aten/cnnl/cnnlOpDescriptors.cpp

namespace oneflow {

void CnnlPoolingDescriptor::set(cnnlPoolingMode_t mode, int kernel_h, int kernel_w, int stride_h,
                                int stride_w, int pad_u, int pad_d, int pad_l, int pad_r) {
  OF_CNNL_CHECK(cnnlSetPooling2dDescriptor(this->mut_desc(), mode, CNNL_PROPAGATE_NAN, kernel_h,
                                           kernel_w, pad_u, pad_d, pad_l, pad_r, stride_h,
                                           stride_w));
}

void CnnlPoolingDescriptor::set(cnnlPoolingMode_t mode, int64_t dims, const int kernel_size[],
                                const int stride[], const int padding[]) {
  OF_CNNL_CHECK(cnnlSetPoolingNdDescriptor(this->mut_desc(), mode, CNNL_NOT_PROPAGATE_NAN, dims,
                                           kernel_size, padding, stride));
}

void CnnlPoolingDescriptor::set(cnnlPoolingMode_t mode, int64_t dims, const int kernel_size[],
                                const int stride[], const int padding[], const int dilation[],
                                bool ceil_mode) {
  OF_CNNL_CHECK(cnnlSetPoolingNdDescriptor_v2(this->mut_desc(), mode, CNNL_NOT_PROPAGATE_NAN, dims,
                                              kernel_size, padding, stride, dilation, ceil_mode));
}

void CnnlTransposeDescriptor::set(const int p_dims, const int permute[]) {
  OF_CNNL_CHECK(cnnlSetTransposeDescriptor(this->mut_desc(), p_dims, permute));
}

void CnnlReduceDescriptor::set(cnnlDataType_t data_type, std::vector<int64_t> axis,
                               cnnlReduceOp_t mode, cnnlReduceIndices_t is_indices,
                               cnnlIndicesType_t indices_type) {
  int axis_num = axis.size();
  std::vector<int> axis_list(axis_num);
  for (int i = 0; i < axis_num; i++) { axis_list[i] = static_cast<int>(axis[i]); }
  OF_CNNL_CHECK(cnnlSetReduceDescriptor(this->mut_desc(), axis_list.data(), axis_num, mode,
                                        data_type, CNNL_NOT_PROPAGATE_NAN, is_indices,
                                        indices_type));
}

void CnnlOpTensorDescriptor::set(cnnlOpTensorDesc_t op_type, cnnlDataType_t op_tensor_comp_type,
                                 cnnlNanPropagation_t op_tensor_nan_opt) {
  OF_CNNL_CHECK(
      cnnlSetOpTensorDescriptor(this->mut_desc(), op_type, op_tensor_comp_type, op_tensor_nan_opt));
}

void CnnlActivationDescriptor::set(cnnlActivationMode_t mode, cnnlNanPropagation_t nanProp,
                                   float ceof) {
  OF_CNNL_CHECK(cnnlSetActivationDescriptor(this->mut_desc(), mode, nanProp, ceof));
}

void CnnlConvolutionDescriptor::set(int dim, int* stride, int* padding, int* dilation,
                                    int64_t groups, cnnlDataType_t dtype) {
  CHECK_GT_OR_THROW(dim, 2) << "Convolution input's dim must greater than 2.";
  int n = dim - 2;
  std::vector<int> padding_t(2 * n);
  std::vector<int> stride_t(n);
  std::vector<int> dilation_t(n);
  int groups_t;
  for (int i = 0; i < n; ++i) {
    padding_t[2 * i] = padding[i];
    padding_t[2 * i + 1] = padding[i];
    stride_t[i] = stride[i];
    dilation_t[i] = dilation[i];
  }
  groups_t = groups;
  OF_CNNL_CHECK(cnnlSetConvolutionDescriptor(this->mut_desc(), dim, padding_t.data(),
                                             stride_t.data(), dilation_t.data(), groups_t, dtype));
}

void CnnlDeconvolutionDescriptor::set(int dim, int* stride, int* padding, int* dilation,
                                      int64_t groups, cnnlDataType_t dtype) {
  CHECK_GT_OR_THROW(dim, 2) << "Convolution input's dim must greater than 2.";
  int n = dim - 2;
  std::vector<int> padding_t(2 * n);
  std::vector<int> stride_t(n);
  std::vector<int> dilation_t(n);
  int groups_t;
  for (int i = 0; i < n; ++i) {
    padding_t[2 * i] = padding[i];
    padding_t[2 * i + 1] = padding[i];
    stride_t[i] = stride[i];
    dilation_t[i] = dilation[i];
  }
  groups_t = groups;
  OF_CNNL_CHECK(cnnlSetDeconvolutionDescriptor(mut_desc(), dim, padding_t.data(), stride_t.data(),
                                               dilation_t.data(), groups_t, dtype));
}

void CnnlUniqueDescriptor::set(bool sorted, int dim, bool return_inverse, bool return_counts) {
  OF_CNNL_CHECK(cnnlSetUniqueDescriptor(this->mut_desc(),
                                        sorted ? CNNL_SORT_ASCEND : CNNL_UNSORT_REVERSE, dim,
                                        return_inverse, return_counts));
}

void CnnlMatmulDescriptor::set_attr(cnnlMatMulDescAttribute_t attr, const void* buf,
                                    size_t size_in_bytes) {
  OF_CNNL_CHECK(cnnlSetMatMulDescAttr(this->mut_desc(), attr, buf, size_in_bytes));
}

void CnnlBatchMatmulDescriptor::set_attr(cnnlBatchMatMulDescAttribute_t attr, const void* buf,
                                         size_t size_in_bytes) {
  OF_CNNL_CHECK(cnnlSetBatchMatMulDescAttr(this->mut_desc(), attr, buf, size_in_bytes));
}

void CnnlCTCLossDescriptor::set(cnnlCTCLossNormalizationMode_t norm_mode,
                                cnnlCTCLossReduceMode_t reduce_mode,
                                cnnlCTCLossZeroInfinityMode_t zero_infinity, int blank,
                                int max_input_length, int max_label_length) {
  OF_CNNL_CHECK(cnnlSetCTCLossDescriptor(this->mut_desc(), norm_mode, reduce_mode, zero_infinity,
                                         blank, max_input_length, max_label_length));
}

void CnnlNmsDescriptor::set(const cnnlNmsOutputMode_t mode, const float iou_threshold,
                            const int max_output_size, const float confidence_threshold,
                            const int input_layout) {
  OF_CNNL_CHECK(cnnlSetNmsDescriptor_v2(this->mut_desc(), mode, iou_threshold, max_output_size,
                                        confidence_threshold, input_layout));
}

void CnnlTrigonDescriptor::set(cnnlTrigonFunctionMode_t mode) {
  OF_CNNL_CHECK(cnnlSetTrigonDescriptor(this->mut_desc(), mode));
}

}  // namespace oneflow
