#include "oneflow/core/common/maybe.h"
#include "oneflow/core/job/job.pb.h"

namespace oneflow {
class NNGraph;
namespace one {
class TensorTuple;
Maybe<void> InterpretJob(const one::TensorTuple& inputs, one::TensorTuple& outputs,
                         const std::shared_ptr<NNGraph>& graph);
}  // namespace one
}  // namespace oneflow
