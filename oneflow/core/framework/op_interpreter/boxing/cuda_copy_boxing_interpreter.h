#ifndef ONEFLOW_CORE_FRAMEWORK_OP_INTERPRETER_BOXING_CUDA_COPY_BOXING_INTERPRETER_H_
#define ONEFLOW_CORE_FRAMEWORK_OP_INTERPRETER_BOXING_CUDA_COPY_BOXING_INTERPRETER_H_

#include "oneflow/core/framework/op_interpreter/boxing/eager_boxing_interpreter.h"

namespace oneflow {

class CudaCopyBoxingInterpreter : public EagerBoxingInterpreter {
 public:
  CudaCopyBoxingInterpreter() = default;
  ~CudaCopyBoxingInterpreter() override = default;

  Maybe<one::Tensor> InterpretImpl(const std::shared_ptr<one::Tensor>& input,
                                   Symbol<cfg::ParallelDistribution> in_nd_sbp,
                                   Symbol<cfg::ParallelDistribution> out_nd_sbp,
                                   Symbol<ParallelDesc> in_parallel_desc,
                                   Symbol<ParallelDesc> out_parallel_desc) const override;
};

}

#endif  // ONEFLOW_CORE_FRAMEWORK_OP_INTERPRETER_BOXING_CUDA_COPY_BOXING_INTERPRETER_H_
