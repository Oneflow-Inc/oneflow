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
#include "oneflow/core/job_rewriter/autotick.h"
#include "oneflow/core/framework/user_op_registry_manager.h"

namespace oneflow {

namespace {

class MutUserOpConTickInputHelper final : public MutOpConTickInputHelper {
 public:
  MutUserOpConTickInputHelper() : MutOpConTickInputHelper() {}

  bool VirtualIsTickInputBound() const override { return !op_conf().user_conf().input().empty(); }

  OperatorConf NewTickInputBoundOpConf(const std::string& lbn) const override {
    OperatorConf ret(op_conf());
    (*ret.mutable_user_conf()->mutable_input())[user_op::kUserSourceOpTickInputArgName].add_s(lbn);
    return ret;
  }
};

}  // namespace

REGISTER_AUTO_TICK(OperatorConf::kUserConf, MutUserOpConTickInputHelper);

}  // namespace oneflow
