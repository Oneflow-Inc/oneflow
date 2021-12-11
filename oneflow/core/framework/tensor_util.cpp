#include "oneflow/core/framework/tensor_util.h"

#include "oneflow/core/common/spin_counter.h"
#include "oneflow/core/framework/instructions_builder.h"

namespace oneflow {
namespace one {

Maybe<void> SyncAccessTensorWithTimeOut(
    const std::shared_ptr<Tensor>& tensor,
    const std::shared_ptr<std::function<void(uint64_t)>>& callback, const std::string& modifier) {
  return SpinCounter::SpinWait(1, [&](const std::shared_ptr<SpinCounter>& sc) -> Maybe<void> {
    return PhysicalRun([&](InstructionsBuilder* builder) -> Maybe<void> {
      return builder->SyncAccessBlobByCallback(JUST(tensor->AsMirroredTensor()), sc, callback,
                                               modifier);
    });
  });
}

}  // namespace one
}  // namespace oneflow
