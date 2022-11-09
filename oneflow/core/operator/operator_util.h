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
#ifndef ONEFLOW_CORE_OPERATOR_OPERATOR_UTIL_H_
#define ONEFLOW_CORE_OPERATOR_OPERATOR_UTIL_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/common/shape_view.h"
#include "oneflow/core/common/protobuf.h"

namespace oneflow {

namespace user_op {

class UserOpConfWrapper;
}

size_t DhwOffset(const std::string& data_format);

std::vector<int32_t> Get3DVecInOpConf(const PbRf<int32_t>& field_vals, int32_t NDims);

int64_t GetInDim(const ShapeView& shape, const std::string& data_format, int32_t dim, int32_t NDim);

void GetWindowedOutputSize(int64_t input_size, int32_t filter_size, int32_t stride,
                           const std::string& padding_type, int64_t* output_size,
                           int32_t* padding_before, int32_t* padding_after);

void GetWindowedOutputSize(int64_t input_size, int32_t filter_size, int32_t stride,
                           const std::string& padding_type, int64_t* output_size,
                           int32_t* padding_size);

void GetWindowedOutputSize(int64_t input_size, int32_t filter_size, int32_t dilation_rate,
                           int32_t stride, const std::string& padding_type, int64_t* output_size,
                           int32_t* padding_before, int32_t* padding_after);

void GetWindowedOutputSize(int64_t input_size, int32_t filter_size, int32_t dilation_rate,
                           int32_t stride, const std::string& padding_type, bool ceil_mode,
                           int64_t* output_size, int32_t* padding_before, int32_t* padding_after);

void Get3DOutputSize(const DimVector& in, const std::vector<int32_t>& pool_size,
                     const std::vector<int32_t>& strides, const std::string& padding_type,
                     DimVector* out, std::vector<int32_t>* padding);

void Get3DOutputSize(const DimVector& in, const std::vector<int32_t>& pool_size,
                     const std::vector<int32_t>& strides, const std::string& padding_type,
                     DimVector* out, std::vector<int32_t>* padding_before,
                     std::vector<int32_t>* padding_after);

void Get3DOutputSize(const DimVector& in, const std::vector<int32_t>& pool_size,
                     const std::vector<int32_t>& strides, const std::string& padding_type,
                     DimVector* out, std::vector<int32_t>* padding_before,
                     std::vector<int32_t>* padding_after, std::vector<int32_t>* dilation_rate);

void Get3DOutputSize(const DimVector& in, const std::vector<int32_t>& pool_size,
                     const std::vector<int32_t>& strides, const std::string& padding_type,
                     const bool ceil_mode, std::vector<int32_t>* dilation_rate, DimVector* out,
                     std::vector<int32_t>* padding_before, std::vector<int32_t>* padding_after);

void GetConvOutAndPad(const ShapeView& in_blob_shape, const PbMessage& conv_conf, DimVector* out,
                      std::vector<int32_t>* pad_small_side, std::vector<int32_t>* pad_large_side);

void GetConvOutAndPad(const ShapeView& in_blob_shape, const user_op::UserOpConfWrapper& conv_conf,
                      DimVector* out, std::vector<int32_t>* pad_small_side,
                      std::vector<int32_t>* pad_large_side);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_OPERATOR_UTIL_H_
