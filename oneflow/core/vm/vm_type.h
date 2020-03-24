#ifndef ONEFLOW_CORE_VM_TYPE_H_
#define ONEFLOW_CORE_VM_TYPE_H_

namespace oneflow {
namespace vm {

enum VmType { kInvalidVmType = 0, kRemote, kLocal };

enum InterpretType { kInvalidInterpretType = 0, kCompute, kInfer };

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_TYPE_H_
