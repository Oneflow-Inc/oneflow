#include "oneflow/core/common/index.h"
#include "oneflow/core/common/protobuf.h"

namespace oneflow {

Index::Index(const std::initializer_list<int64_t>& dim_vec) : dim_vec_(dim_vec) {}

Index::Index(const std::vector<int64_t>& dim_vec) : dim_vec_(dim_vec) {}

Index& Index::operator=(const Index& shape) {
  dim_vec_ = shape.dim_vec_;
  return *this;
}

bool Index::operator==(const Index& rhs) const { return dim_vec_ == rhs.dim_vec_; }

}  // namespace oneflow
