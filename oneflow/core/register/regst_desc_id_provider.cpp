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
#include "oneflow/core/register/regst_desc_id_provider.h"
#include <glog/logging.h>
#include "oneflow/core/job/id_manager.h"
#include "oneflow/core/job/compile_mode.h"

namespace oneflow {

ConstRegstDescIdProvider::ConstRegstDescIdProvider() {
  regst_desc_id_ = Singleton<IDMgr>::Get()->NewRegstDescId();
}

int64_t LazyInitRegstDescIdProvider::regst_desc_id() const {
  int64_t ret = regst_desc_id_.load(std::memory_order_acquire);
  CHECK_NE(ret, 0);
  return ret;
}

void LazyInitRegstDescIdProvider::init_regst_desc_id() {
  CHECK_EQ(regst_desc_id_, 0);
  regst_desc_id_ = Singleton<IDMgr>::Get()->NewRegstDescId();
}

void LazyInitRegstDescIdProvider::init_regst_desc_id(int64_t regst_desc_id) {
  CHECK_NE(regst_desc_id, 0);
  CHECK_EQ(regst_desc_id_, 0);
  regst_desc_id_ = regst_desc_id;
}

namespace {

struct RawNewRegstDescIdProvider final : public CompileModeVisitor<RawNewRegstDescIdProvider> {
  static std::unique_ptr<RegstDescIdProvider> VisitNaive() {
    return std::make_unique<ConstRegstDescIdProvider>();
  }
  static std::unique_ptr<RegstDescIdProvider> VisitRankPerIter() {
    return std::make_unique<LazyInitRegstDescIdProvider>();
  }
  static std::unique_ptr<RegstDescIdProvider> VisitRankPerThread() {
    return std::make_unique<LazyInitRegstDescIdProvider>();
  }
};

}  // namespace

std::unique_ptr<RegstDescIdProvider> NewRegstDescIdProvider() {
  return RawNewRegstDescIdProvider::Visit(CHECK_JUST(CurrentCompileMode()));
}

}  // namespace oneflow
