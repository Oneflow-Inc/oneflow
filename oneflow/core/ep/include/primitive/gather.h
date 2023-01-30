#ifndef ONEFLOW_CORE_EP_PRIMITIVE_GATHER_H
#define ONEFLOW_CORE_EP_PRIMITIVE_GATHER_H

#include "oneflow/core/ep/include/primitive/primitive.h"
#include "oneflow/core/ep/include/primitive/blas.h"
#include "oneflow/core/common/scalar.h"

namespace oneflow {

namespace ep {

namespace primitive {

class Gather : public Primitive {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Gather);
  Gather() = default;
  ~Gather() override = default;
  
  virtual void Launch(
      Stream *stream,
      const void *indices,
      int64_t num_indices,
      const void* in,
      const int64_t outer_dim_size,
      const int64_t gather_dim_size,
      const int64_t inner_dim_size,
      void* out,
      const int64_t offset) = 0;
};

class GatherFactory : public Factory<Gather>{
 public:
  OF_DISALLOW_COPY_AND_MOVE(GatherFactory);
  GatherFactory() = default;
  ~GatherFactory() override = default;

  virtual std::unique_ptr<Gather> New(DataType params_type, DataType indices_type) = 0;
};

} // namespace primitive

} // namespace ep

} // namespace oneflow

#endif // ONEFLOW_CORE_EP_PRIMITIVE_GATHER_H
