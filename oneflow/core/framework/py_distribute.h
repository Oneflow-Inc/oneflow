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
#ifndef ONEFLOW_CORE_FRAMEWORK_PY_DISTRIBUTE_H_
#define ONEFLOW_CORE_FRAMEWORK_PY_DISTRIBUTE_H_

#include "oneflow/core/common/maybe.h"
#include "oneflow/core/job/sbp_parallel.cfg.h"

namespace oneflow {

namespace compatible_py {

static const int64_t HAS_NO_AXIS = -1;

class Distribute {
 public:
  Distribute() : sbp_parallel_(std::make_shared<cfg::SbpParallel>()) {}
  Distribute(const Distribute&) = delete;
  Distribute(Distribute&&) = delete;
  virtual ~Distribute() = default;

  virtual int64_t axis() const { return HAS_NO_AXIS; }

 protected:
  std::shared_ptr<cfg::SbpParallel> sbp_parallel_;
};

class AutoDistribute : public Distribute {
 public:
  AutoDistribute() : Distribute() {}
  AutoDistribute(const AutoDistribute&) = delete;
  AutoDistribute(AutoDistribute&&) = delete;
  ~AutoDistribute() override = default;
};

class BroadcastDistribute : public Distribute {
 public:
  BroadcastDistribute() : Distribute() { sbp_parallel_->mutable_broadcast_parallel(); }
  BroadcastDistribute(const BroadcastDistribute&) = delete;
  BroadcastDistribute(BroadcastDistribute&&) = delete;
  ~BroadcastDistribute() override = default;
};

class SplitDistribute : public Distribute {
 public:
  SplitDistribute(int axis) : Distribute() {
    sbp_parallel_->mutable_split_parallel()->set_axis(axis);
  }
  SplitDistribute(const SplitDistribute&) = delete;
  SplitDistribute(SplitDistribute&&) = delete;
  ~SplitDistribute() override = default;

  int64_t axis() const override { return sbp_parallel_->split_parallel().axis(); }
};

std::shared_ptr<AutoDistribute> GlobalAutoDistribute();
std::shared_ptr<BroadcastDistribute> GlobalBroadcastDistribute();
Maybe<SplitDistribute> GlobalSplitDistribute(int axis);

Maybe<Distribute> MakeDistribute(const cfg::SbpParallel& sbp_parallel);

}  // namespace compatible_py

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_PY_DISTRIBUTE_H_
