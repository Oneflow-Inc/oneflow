#ifndef ONEFLOW_CORE_OPERATOR_OPERATOR_UTIL_H_
#define ONEFLOW_CORE_OPERATOR_OPERATOR_UTIL_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/common/shape_view.h"
#include "oneflow/core/common/protobuf.h"

namespace oneflow {

const size_t DhwOffset(const std::string& data_format);

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

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_OPERATOR_UTIL_H_
