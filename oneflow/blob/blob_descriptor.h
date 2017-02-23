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
  
  void init(const Shape& rhs_shape,
            const MemoryContext& rhs_memory_context,
            FloatType rhs_float_type) {
    shape_ = rhs_shape;
    memory_context_ = rhs_memory_context;
    float_type_ = rhs_float_type;
  }

  const Shape& shape() const { return shape_; }
  const MemoryContext& memory_context() const { return memory_context_; }
  size_t byte_size() const {
    return shape_.elem_cnt() * GetFloatByteSize(float_type_);
  }
 
 private:
  Shape shape_;
  MemoryContext memory_context_;
  FloatType float_type_;

};

} // namespace oneflow

#endif // ONEFLOW_BLOB_DESCRIPTOR_H_
