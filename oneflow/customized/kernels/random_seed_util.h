#ifndef ONEFLOW_CUSTOMIZED_KERNELS_RANDOM_SEED_UTIL_H_
#define ONEFLOW_CUSTOMIZED_KERNELS_RANDOM_SEED_UTIL_H_

#include "oneflow/core/framework/op_kernel.h"

namespace oneflow {

int64_t GetOpKernelRandomSeed(const user_op::KernelInitContext* ctx);

}  // namespace oneflow

#endif  // ONEFLOW_CUSTOMIZED_KERNELS_RANDOM_SEED_UTIL_H_
