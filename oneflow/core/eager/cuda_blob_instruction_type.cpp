#include "oneflow/core/eager/blob_instruction_type.h"
#include "oneflow/core/vm/cuda_stream_type.h"

namespace oneflow {
namespace eager {
class GpuLazyReferenceInstructionType : public LazyReferenceInstructionType {
 public:
  GpuLazyReferenceInstructionType() = default;
  ~GpuLazyReferenceInstructionType() override = default;

  using stream_type = vm::CudaStreamType;
};

COMMAND(vm::RegisterInstructionType<GpuLazyReferenceInstructionType>("gpu.LazyReference"));

}  // namespace eager
}  // namespace oneflow
