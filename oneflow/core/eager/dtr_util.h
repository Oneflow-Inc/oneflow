#include <memory>
#include <vector>

#include "oneflow/core/common/maybe.h"

namespace oneflow {

bool DTREnabled();
size_t GetDTRMemoryThreshold();
bool DTRDebugEnabled();
int DTRDebugLevel();
bool dtr_use_disjoint_set();

namespace vm {

class DTREagerBlobObject;
class LocalCallOpKernelPhyInstrOperand;
class DTRInstrOperand;

std::vector<std::shared_ptr<DTREagerBlobObject>> GetDTRInputs(
    const LocalCallOpKernelPhyInstrOperand* operand);
std::vector<std::shared_ptr<DTREagerBlobObject>> GetDTROutputs(
    const LocalCallOpKernelPhyInstrOperand* operand);

std::vector<std::shared_ptr<DTREagerBlobObject>> GetDTRInputs(
    const std::shared_ptr<const LocalCallOpKernelPhyInstrOperand>& operand);
std::vector<std::shared_ptr<DTREagerBlobObject>> GetDTROutputs(
    const std::shared_ptr<const LocalCallOpKernelPhyInstrOperand>& operand);

std::shared_ptr<LocalCallOpKernelPhyInstrOperand> DTROp2LocalCallOp(DTRInstrOperand* operand);

Maybe<void> CheckInputInMemory(LocalCallOpKernelPhyInstrOperand* operand);
Maybe<void> CheckOutputInMemory(LocalCallOpKernelPhyInstrOperand* operand);

}  // namespace vm
}  // namespace oneflow
