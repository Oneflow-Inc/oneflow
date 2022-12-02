#include "oneflow/api/python/dlpack/dlpack.h"
#include "oneflow/core/common/maybe.h"

namespace oneflow {

namespace one {
class Tensor;
}

Maybe<one::Tensor> fromDLPack(const DLManagedTensor* src);

}  // namespace oneflow
