#ifndef ONEFLOW_CORE_COMMON_ND_INDEX_H_
#define ONEFLOW_CORE_COMMON_ND_INDEX_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/common/shape.h"

namespace oneflow {

class NdIndex final {
 public:
  NdIndex() = default;
  explicit NdIndex(const DimVector& dim_vec);
  NdIndex(const std::initializer_list<int64_t>& dim_vec);
  ~NdIndex() = default;
  NdIndex& operator=(const NdIndex& other);

  bool operator==(const NdIndex& rhs) const;
  bool operator!=(const NdIndex& rhs) const { return !(*this == rhs); }

  int64_t At(int64_t index) const { return dim_vec_.at(index); }
  int64_t NumAxes() const { return dim_vec_.size(); }

 private:
  DimVector dim_vec_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_ND_INDEX_H_
