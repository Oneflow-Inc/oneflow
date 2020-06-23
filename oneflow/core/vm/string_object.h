#ifndef ONEFLOW_CORE_VM_STRING_OBJECT_H_
#define ONEFLOW_CORE_VM_STRING_OBJECT_H_

#include <string>
#include "oneflow/core/vm/object.h"

namespace oneflow {
namespace vm {

class StringObject final : public Object {
 public:
  StringObject(const StringObject&) = delete;
  StringObject(StringObject&&) = delete;

  StringObject(const std::string& str) : str_(str) {}
  ~StringObject() override = default;

  const std::string& str() const { return str_; }

 private:
  std::string str_;
};

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_STRING_OBJECT_H_
