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
#ifndef ONEFLOW_CORE_FRAMEWORK_ND_SBP_H_
#define ONEFLOW_CORE_FRAMEWORK_ND_SBP_H_

#include <vector>
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/symbol.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/common/decorator.h"
#include "oneflow/core/job/sbp_parallel.h"

namespace oneflow {

Maybe<Symbol<NdSbp>> GetDualNdSbp(Symbol<NdSbp> nd_sbp);

Maybe<Symbol<NdSbp>> GetDualNdSbp(Symbol<NdSbp> sbp_list);

Maybe<std::vector<std::string>> GetNdSbpStrList(const std::vector<Symbol<SbpParallel>>& sbp_list);

Maybe<std::vector<std::string>> GetNdSbpStrList(Symbol<NdSbp> nd_sbp);

Maybe<std::vector<std::string>> GetDualNdSbpStrList(Symbol<NdSbp> nd_sbp);

Maybe<std::vector<std::string>> GetDualNdSbpStrList(Symbol<NdSbp> nd_sbp);

namespace private_details {

Maybe<Symbol<NdSbp>> RawGetNdSbp(const std::vector<Symbol<SbpParallel>>& sbp_list);
Maybe<std::vector<Symbol<SbpParallel>>> RawGetSbpList(Symbol<NdSbp> nd_sbp);
bool RawContainSplitSbp(Symbol<NdSbp> nd_sbp);

Maybe<std::vector<Symbol<SbpParallel>>> RawNdSbpReplacePartialByBroadcast(
    const std::vector<Symbol<SbpParallel>>& sbp_list);

}  // namespace private_details

static constexpr auto* GetNdSbp = DECORATE(&private_details::RawGetNdSbp, ThreadLocalCopiable);
static constexpr auto* GetSbpList = DECORATE(&private_details::RawGetSbpList, ThreadLocal);
static constexpr auto* ContainSplitSbp =
    DECORATE(&private_details::RawContainSplitSbp, ThreadLocal);
const std::vector<Symbol<SbpParallel>>& GetNoneSbpList();

static constexpr auto* NdSbpReplacePartialByBroadcast =
    DECORATE(&private_details::RawNdSbpReplacePartialByBroadcast, ThreadLocalCachedCopiable);

std::string SbpToString(Symbol<SbpParallel> sbp_sym);
std::string NdSbpToString(Symbol<NdSbp> nd_sbp_sym);
std::string SbpToString(const SbpParallel& sbp);
std::string NdSbpToString(const NdSbp& nd_sbp);

Maybe<Symbol<NdSbp>> SetSbpAtAxis(Symbol<NdSbp> nd_sbp, Symbol<SbpParallel> sbp, int axis);
Maybe<Symbol<NdSbp>> SetSbpAtAxis(const NdSbp& nd_sbp, const SbpParallel& sbp, int axis);

Maybe<Symbol<NdSbp>> SbpToNdSbp(Symbol<SbpParallel> sbp);
Maybe<Symbol<NdSbp>> SbpToNdSbp(const SbpParallel& sbp);

// If an nd sbp can be converted to a 1d sbp.
bool Is1dSbp(const NdSbp& nd_sbp);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_ND_SBP_H_
