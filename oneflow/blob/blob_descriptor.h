#ifndef ONEFLOW_BLOB_DESCRIPTOR_H_
#define ONEFLOW_BLOB_DESCRIPTOR_H_

#include "blob/shape.h"
#include "common/util.h"
#include "memory/memory_context.h"

namespace oneflow {

class BlobDescriptor {
 public:
  DISALLOW_COPY_AND_MOVE(BlobDescriptor);
  BlobDescriptor() = default;
  ~BlobDescriptor() = default;
  
  void Init() {
    // struct style
  }

  const Shape& shape() const { return shape_; }
  const MemoryContext& memory_context() const { return memory_context_; }
  size_t ByteSize() const {
    return shape_.elem_cnt() * GetFloatByteSize(float_type_);
  }

  Shape& mutable_shape() { return shape_; }
  MemoryContext& mutable_memory_context() { return memory_context_; }
  FloatType& mutable_float_type() { return float_type_; }
 
 private:
  Shape shape_;
  MemoryContext memory_context_;
  FloatType float_type_;

};

} // namespace oneflow

#endif // ONEFLOW_BLOB_DESCRIPTOR_H_
