#include <memory>
#include <vector>

#include "oneflow/core/common/env_var/dtr.h"
#include "oneflow/core/common/maybe.h"

namespace oneflow {

namespace dtr {

bool is_enabled();
size_t memory_threshold();
bool is_enabled_and_debug();
int debug_level();
bool is_using_disjoint_set();
bool is_check_enabled();

}  // namespace dtr

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
