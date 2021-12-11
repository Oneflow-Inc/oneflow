#include <string>

#include "oneflow/core/common/maybe.h"

namespace oneflow {
namespace one {

class Tensor;

Maybe<void> SyncAccessTensorWithTimeOut(
    const std::shared_ptr<Tensor>& tensor,
    const std::shared_ptr<std::function<void(uint64_t)>>& callback, const std::string& modifier);
}
}
