#ifndef ONEFLOW_CORE_FRAMEWORK_OP_INTERPRETER_BOXING_CUDA_BASED_CPU_MPI_BOXING_INTERPRETER_H_
#define ONEFLOW_CORE_FRAMEWORK_OP_INTERPRETER_BOXING_CUDA_BASED_CPU_MPI_BOXING_INTERPRETER_H_

#include "oneflow/core/framework/boxing/eager_boxing_interpreter.h"

namespace oneflow {

class CudaBasedCpuMpiBoxingInterpreter final : public EagerBoxingInterpreter {
 public:
  explicit CudaBasedCpuMpiBoxingInterpreter(const std::shared_ptr<EagerBoxingInterpreter>& base_gpu_boxing_interpreter) : base_gpu_boxing_interpreter_(base_gpu_boxing_interpreter) {}

  ~CudaBasedCpuMpiBoxingInterpreter() override = default;

 private:
  Maybe<one::Tensor> InterpretImpl(const std::shared_ptr<one::Tensor>& input,
                                   Symbol<cfg::ParallelDistribution> in_nd_sbp,
                                   Symbol<cfg::ParallelDistribution> out_nd_sbp,
                                   Symbol<ParallelDesc> in_parallel_desc,
                                   Symbol<ParallelDesc> out_parallel_desc) const override;

  std::shared_ptr<EagerBoxingInterpreter> base_gpu_boxing_interpreter_;
}

#endif  // ONEFLOW_CORE_FRAMEWORK_OP_INTERPRETER_BOXING_CUDA_BASED_CPU_MPI_BOXING_INTERPRETER_H_
