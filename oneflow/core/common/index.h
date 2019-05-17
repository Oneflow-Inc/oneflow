#ifndef ONEFLOW_CORE_COMMON_INDEX_H_
#define ONEFLOW_CORE_COMMON_INDEX_H_

#include "oneflow/core/common/util.h"

namespace oneflow {

class Index final {
 public:
  explicit Index(const std::vector<int64_t>& dim_vec);
  Index(const std::initializer_list<int64_t>& dim_vec);
  ~Index() = default;
  Index& operator=(const Index& other);

  bool operator==(const Index& rhs) const;
  bool operator!=(const Index& rhs) const { return !(*this == rhs); }

  int64_t At(int64_t index) const { return dim_vec_[index]; }
  int64_t NumAxes() const { return dim_vec_.size(); }

  std::vector<int64_t> dim_vec_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_INDEX_H_
