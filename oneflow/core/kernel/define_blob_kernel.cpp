#include "oneflow/core/kernel/define_blob_kernel.h"

namespace oneflow {

REGISTER_KERNEL(OperatorConf::kDefineBlobConf, DefineBlobKernel);
}
