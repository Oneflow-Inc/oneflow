#ifndef ONEFLOW_CORE_VM_INSTRUCTION_TRAIT_H_
#define ONEFLOW_CORE_VM_INSTRUCTION_TRAIT_H_

#include "oneflow/core/vm/cuda_stream_type.h"

namespace oneflow {
namespace vm {

class CudaStreamInstructionTrait {
 public:
  virtual ~CudaStreamInstructionTrait() = default;

  using stream_type = CudaStreamType;

  virtual bool do_event_record() const = 0;
};

}
}

#endif  // ONEFLOW_CORE_VM_INSTRUCTION_TRAIT_H_
