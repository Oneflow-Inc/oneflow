#include "oneflow/core/common/nd_index.h"
#include "oneflow/core/common/protobuf.h"

namespace oneflow {

NdIndex::NdIndex(const std::initializer_list<int64_t>& dim_vec) : dim_vec_(dim_vec) {}

NdIndex::NdIndex(const DimVector& dim_vec) : dim_vec_(dim_vec) {}

NdIndex& NdIndex::operator=(const NdIndex& shape) {
  dim_vec_ = shape.dim_vec_;
  return *this;
}

bool NdIndex::operator==(const NdIndex& rhs) const { return dim_vec_ == rhs.dim_vec_; }

}  // namespace oneflow
