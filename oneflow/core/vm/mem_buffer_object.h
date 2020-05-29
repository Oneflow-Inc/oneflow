#ifndef ONEFLOW_CORE_VM_MEM_BUFFER_OBJECT_H_
#define ONEFLOW_CORE_VM_MEM_BUFFER_OBJECT_H_

#include "oneflow/core/vm/object.h"
#include "oneflow/core/memory/memory_case.pb.h"

namespace oneflow {
namespace vm {

class MemBufferObjectType final : public Object {
 public:
  MemBufferObjectType(const MemBufferObjectType&) = delete;
  MemBufferObjectType(MemBufferObjectType&) = delete;

  MemBufferObjectType() = default;
  ~MemBufferObjectType() override = default;

  const MemoryCase& mem_case() const { return mem_case_; }
  std::size_t size() const { return size_; }

  MemoryCase* mut_mem_case() { return &mem_case_; }
  void set_size(std::size_t val) { size_ = val; }

 private:
  MemoryCase mem_case_;
  std::size_t size_;
};

class MemBufferObjectValue final : public Object {
 public:
  MemBufferObjectValue(const MemBufferObjectValue&) = delete;
  MemBufferObjectValue(MemBufferObjectValue&) = delete;

  MemBufferObjectValue() = default;
  ~MemBufferObjectValue() override = default;

  const char* data() const { return data_; }
  char* mut_data() { return data_; }
  void reset_data(char* val) { data_ = val; }

 private:
  char* data_;
};

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_MEM_BUFFER_OBJECT_H_
