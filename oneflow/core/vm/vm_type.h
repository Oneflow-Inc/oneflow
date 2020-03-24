#ifndef ONEFLOW_CORE_VM_TYPE_H_
#define ONEFLOW_CORE_VM_TYPE_H_

namespace oneflow {
namespace vm {

enum VmType { kRemote = 0, kLocal };

enum InterpretType { kCompute = 0, kInfer };

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_TYPE_H_
