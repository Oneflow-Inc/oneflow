#ifndef ONEFLOW_CORE_FRAMEWORK_PLACED_ND_SBP_H_
#define ONEFLOW_CORE_FRAMEWORK_PLACED_ND_SBP_H_

#include <functional>
#include "oneflow/core/common/symbol.h"
#include "oneflow/core/common/maybe.h"

namespace oneflow {

namespace cfg {
class NdSbp;
}
class ParallelDesc;

class PlacedNdSbp final {
 public:
  PlacedNdSbp(const Symbol<cfg::NdSbp>& nd_sbp, const Symbol<ParallelDesc>& placement)
    : nd_sbp_(nd_sbp), placement_(placement) {}
  ~PlacedNdSbp() = default;

  static Maybe<Symbol<PlacedNdSbp>> (*New)(const Symbol<cfg::NdSbp>&, const Symbol<ParallelDesc>&);

  const Symbol<cfg::NdSbp>& nd_sbp() const { return nd_sbp_; } 
  const Symbol<ParallelDesc>& placement() const { return placement_; }
  
  bool operator==(const PlacedNdSbp& other) const {
    return this->nd_sbp_ == other.nd_sbp_ && this->placement_ == other.placement_;
  }

 private:
  Symbol<cfg::NdSbp> nd_sbp_;
  Symbol<ParallelDesc> placement_;
};

}

namespace std {

template<>
struct hash<oneflow::PlacedNdSbp> final {
  size_t operator()(const oneflow::PlacedNdSbp& placed_nd_sbp) const {
    return std::hash<Symbol<cfg::NdSbp>>()(placed_nd_sbp.nd_sbp())
        ^ std::hash<Symbol<ParallelDesc>>()(placed_nd_sbp.placement());
  }
};

}

#endif  // ONEFLOW_CORE_FRAMEWORK_PLACED_ND_SBP_H_
