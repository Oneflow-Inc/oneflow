/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include "oneflow/core/framework/placed_nd_sbp.h"
#include "oneflow/core/job/sbp_parallel.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/common/decorator.h"

namespace oneflow {

namespace {

Maybe<Symbol<PlacedNdSbp>> RawNew(const Symbol<NdSbp>& nd_sbp,
                                  const Symbol<ParallelDesc>& placement) {
  CHECK_OR_RETURN(nd_sbp);
  CHECK_OR_RETURN(placement);
  CHECK_GT_OR_RETURN(nd_sbp->sbp_parallel_size(), 0);
  CHECK_EQ_OR_RETURN(nd_sbp->sbp_parallel_size(), placement->hierarchy()->NumAxes());
  return SymbolOf(PlacedNdSbp(nd_sbp, placement));
}

}  // namespace

decltype(PlacedNdSbp::New) PlacedNdSbp::New = DECORATE(&RawNew, ThreadLocal);

}  // namespace oneflow
