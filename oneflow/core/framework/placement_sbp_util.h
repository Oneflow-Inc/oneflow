#ifndef ONEFLOW_CORE_FRAMEWORK_PLACEMENT_SBP_UTIL_H_
#define ONEFLOW_CORE_FRAMEWORK_PLACEMENT_SBP_UTIL_H_

#include "oneflow/core/common/maybe.h"
#include "oneflow/core/common/symbol.h"

namespace oneflow {

class Shape;
class ParallelDesc;

namespace cfg {

class ParallelDistribution;

}

Maybe<Symbol<ParallelDesc>> GetBroadcastSubParallelDesc(
    Symbol<ParallelDesc> parallel_desc, Symbol<cfg::ParallelDistribution> parallel_distribution);

Maybe<std::vector<int64_t>> GetBroadcastParallelIds(
    const Shape& hierarchy_shape, const std::vector<bool>& dim2is_broadcast, int64_t parallel_id);

}

#endif  // ONEFLOW_CORE_FRAMEWORK_PLACEMENT_SBP_UTIL_H_
