#include "oneflow/xrt/tvm/ops/tvm_op_kernel.h"

namespace oneflow {
namespace xrt {
namespace of_tvm {

class BatchNormOp final : public TVMOpKernel {
 public:
  void Compile(TVMOpContext* ctx) override {
    //TODO
  }
};

// REGISTER_TVM_OP_KERNEL(BatchNorm, BatchNormOp).EnableTrainPhase().Finalize();

}
}
}
