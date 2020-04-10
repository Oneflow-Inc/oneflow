#include "oneflow/core/common/util.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/eager/opkernel_object.h"
#include "oneflow/core/eager/blob_object.h"
#include "oneflow/core/eager/opkernel_instruction.msg.h"
#include "oneflow/core/eager/opkernel_instruction_type.h"
#include "oneflow/core/vm/string_object.h"
#include "oneflow/core/vm/stream.msg.h"
#include "oneflow/core/vm/cuda_stream_type.h"
#include "oneflow/core/vm/instruction.msg.h"
#include "oneflow/core/vm/object.h"

namespace oneflow {
namespace eager {

class CudaCallOpKernelInstructionType final : public CallOpKernelInstructionType {
 public:
  CudaCallOpKernelInstructionType() = default;
  ~CudaCallOpKernelInstructionType() override = default;

  using stream_type = vm::CudaStreamType;
};
COMMAND(vm::RegisterInstructionType<CudaCallOpKernelInstructionType>("CudaCallOpKernel"));

}  // namespace eager
}  // namespace oneflow
