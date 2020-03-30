#ifndef ONEFLOW_CORE_VM_OBJECT_H_
#define ONEFLOW_CORE_VM_OBJECT_H_

namespace oneflow {
namespace vm {

class Object {
 public:
  virtual ~Object() = default;

 protected:
  Object() = default;
};

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_OBJECT_H_
