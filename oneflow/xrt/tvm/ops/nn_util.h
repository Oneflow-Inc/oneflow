#ifndef ONEFLOW_XRT_TVM_OPS_NN_UTIL_H_
#define ONEFLOW_XRT_TVM_OPS_NN_UTIL_H_

#include "oneflow/xrt/tvm/ops/tvm_op_kernel.h"

namespace oneflow {
namespace xrt {
namespace of_tvm {

tvm::Array<tvm::relay::IndexExpr> Calc2DPadding(const std::string& padding_format,
    const std::vector<int32_t>& input_size, const std::vector<int32_t>& filter_size,
    const std::vector<int32_t>& stride, const std::vector<int32_t>& dilation);

tvm::Array<tvm::relay::IndexExpr> Calc2DPadding4Pool(const std::string& data_format,
    const std::string& padding_format, const Shape& in_shape,
    const std::vector<int32_t>& pool_size, const std::vector<int32_t>& stride);

}
}
}

#endif // ONEFLOW_XRT_TVM_OPS_NN_UTIL_H_
