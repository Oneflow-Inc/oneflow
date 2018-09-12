#include "oneflow/core/kernel/shared_model_diff_add_kernel.h"

namespace oneflow {

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kSharedModelDiffAddConf, SharedModelDiffAddKernel,
                           ARITHMETIC_DATA_TYPE_SEQ);

}  // namespace oneflow
