#include "oneflow/core/eager/blob_instruction_type.h"
#include "oneflow/core/vm/cpu_stream_type.h"

namespace oneflow {
namespace eager {
class CpuLazyReferenceInstructionType : public LazyReferenceInstructionType {
 public:
  CpuLazyReferenceInstructionType() = default;
  ~CpuLazyReferenceInstructionType() override = default;

  using stream_type = vm::CpuStreamType;
};

COMMAND(vm::RegisterInstructionType<CpuLazyReferenceInstructionType>("cpu.LazyReference"));

}  // namespace eager
}  // namespace oneflow
