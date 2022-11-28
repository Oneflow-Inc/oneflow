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
#include "oneflow/core/job/rank_group_scope.h"

namespace oneflow {

namespace {

const RankGroupScope** MutThreadLocalRankGroupScope() {
  static thread_local const RankGroupScope* scope = nullptr;
  return &scope;
}

}  // namespace

RankGroupScope::RankGroupScope(Symbol<RankGroup> rank_group, const RankGroupScope* parent,
                               const RankGroupScope* root)
    : rank_group_(rank_group), parent_(parent), root_(root) {
  CHECK_EQ(parent, *MutThreadLocalRankGroupScope());
  *MutThreadLocalRankGroupScope() = this;
}

Maybe<void> RankGroupScope::SetRootSelf() {
  CHECK_ISNULL_OR_RETURN(parent_);
  CHECK_ISNULL_OR_RETURN(root_);
  root_ = this;
  return Maybe<void>::Ok();
}

RankGroupScope::~RankGroupScope() {
  CHECK_EQ(this, *MutThreadLocalRankGroupScope());
  *MutThreadLocalRankGroupScope() = this->parent_;
}

/*static*/ Maybe<RankGroupScope> RankGroupScope::MakeInitialRankGroupScope(
    Symbol<RankGroup> rank_group) {
  CHECK_ISNULL_OR_RETURN(*MutThreadLocalRankGroupScope());
  auto* ptr = new RankGroupScope(rank_group, nullptr, nullptr);
  JUST(ptr->SetRootSelf());
  return std::shared_ptr<RankGroupScope>(ptr);
}

/*static*/ Maybe<RankGroupScope> RankGroupScope::MakeNestedRankGroupScope(
    Symbol<RankGroup> rank_group) {
  const auto* parent = *MutThreadLocalRankGroupScope();
  CHECK_NOTNULL_OR_RETURN(parent);
  const auto* root = &parent->root();
  auto* ptr = new RankGroupScope(rank_group, parent, root);
  return std::shared_ptr<RankGroupScope>(ptr);
}

/*static*/ Maybe<Symbol<RankGroup>> RankGroupScope::CurrentRankGroup() {
  const RankGroupScope* scope = *MutThreadLocalRankGroupScope();
  CHECK_NOTNULL_OR_RETURN(scope);
  return scope->rank_group();
}

/*static*/ Maybe<Symbol<RankGroup>> RankGroupScope::RootRankGroup() {
  const RankGroupScope* scope = *MutThreadLocalRankGroupScope();
  CHECK_NOTNULL_OR_RETURN(scope);
  return scope->root().rank_group();
}

}  // namespace oneflow
