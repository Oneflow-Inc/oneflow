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

std::vector<std::shared_ptr<SplitSbpDescription>> MakeSplitSbpDescriptions(int64_t axis_num_max) {
  std::vector<std::shared_ptr<SplitSbpDescription>> ret(axis_num_max);
  for (int i = 0; i < axis_num_max; ++i) { ret[i].reset(new SplitSbpDescription(i)); }
  return ret;
}

}  // namespace

std::shared_ptr<AutoSbpDescription> g_auto(new AutoSbpDescription());

std::shared_ptr<BroadcastSbpDescription> g_broadcast(new BroadcastSbpDescription());

std::vector<std::shared_ptr<SplitSbpDescription>> g_split(MakeSplitSbpDescriptions(kAxisNumMax));

std::shared_ptr<AutoSbpDescription> GlobalAutoSbpDescription() { return g_auto; }

std::shared_ptr<BroadcastSbpDescription> GlobalBroadcastSbpDescription() { return g_broadcast; }

Maybe<SplitSbpDescription> GlobalSplitSbpDescription(int axis) {
  return JUST(VectorAt(g_split, axis));
}

}  // namespace compatible_py

}  // namespace oneflow