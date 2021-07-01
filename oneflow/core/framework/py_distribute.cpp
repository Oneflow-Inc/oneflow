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
#include "oneflow/core/common/container_util.h"
#include "oneflow/core/framework/py_distribute.h"

namespace oneflow {

namespace compatible_py {

static const int64_t kAxisNumMax = 20;

namespace {

std::vector<std::shared_ptr<SplitDistribute>> MakeSplitDistributes(int64_t axis_num_max) {
  std::vector<std::shared_ptr<SplitDistribute>> ret(axis_num_max);
  for (int i = 0; i < axis_num_max; ++i) { ret[i].reset(new SplitDistribute(i)); }
  return ret;
}

}  // namespace

std::shared_ptr<AutoDistribute> g_auto(new AutoDistribute());

std::shared_ptr<BroadcastDistribute> g_broadcast(new BroadcastDistribute());

std::vector<std::shared_ptr<SplitDistribute>> g_split(MakeSplitDistributes(kAxisNumMax));

std::shared_ptr<AutoDistribute> GlobalAutoDistribute() { return g_auto; }

std::shared_ptr<BroadcastDistribute> GlobalBroadcastDistribute() { return g_broadcast; }

Maybe<SplitDistribute> GlobalSplitDistribute(int axis) { return JUST(VectorAt(g_split, axis)); }

Maybe<Distribute> MakeDistribute(const cfg::SbpParallel& sbp_parallel) {
  if (sbp_parallel.has_broadcast_parallel()) {
    return std::shared_ptr<Distribute>(GlobalBroadcastDistribute());
  } else if (sbp_parallel.has_split_parallel()) {
    auto split_distribute = JUST(GlobalSplitDistribute(sbp_parallel.split_parallel().axis()));
    return std::shared_ptr<Distribute>(split_distribute);
  } else {
    return std::shared_ptr<Distribute>(GlobalAutoDistribute());
  }
}

}  // namespace compatible_py

}  // namespace oneflow
