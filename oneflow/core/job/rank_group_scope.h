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
#ifndef ONEFLOW_CORE_JOB_RANK_GROUP_SCOPE_H_
#define ONEFLOW_CORE_JOB_RANK_GROUP_SCOPE_H_

#include "oneflow/core/job/rank_group.h"
#include "oneflow/core/common/symbol.h"

namespace oneflow {

class RankGroupScope final {
 public:
  ~RankGroupScope();

  Symbol<RankGroup> rank_group() const { return rank_group_; }
  const RankGroupScope& root() const { return *root_; }

  static Maybe<RankGroupScope> MakeNestedRankGroupScope(Symbol<RankGroup> rank_group);

  static Maybe<RankGroupScope> MakeInitialRankGroupScope(Symbol<RankGroup> rank_group);

  static Maybe<Symbol<RankGroup>> CurrentRankGroup();

  static Maybe<Symbol<RankGroup>> RootRankGroup();

 private:
  RankGroupScope(Symbol<RankGroup> rank_group, const RankGroupScope* parent,
                 const RankGroupScope* root);

  Maybe<void> SetRootSelf();

  Symbol<RankGroup> rank_group_;
  const RankGroupScope* parent_;
  const RankGroupScope* root_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_RANK_GROUP_SCOPE_H_
