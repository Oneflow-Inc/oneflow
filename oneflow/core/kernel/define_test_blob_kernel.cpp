#include "oneflow/core/kernel/define_test_blob_kernel.h"

namespace oneflow {

REGISTER_KERNEL(OperatorConf::kDefineTestBlobConf, DefineTestBlobKernel);

}  // namespace oneflow
