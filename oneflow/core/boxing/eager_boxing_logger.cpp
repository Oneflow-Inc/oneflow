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
#include "oneflow/core/common/singleton.h"
#include "oneflow/core/common/decorator.h"
#include "oneflow/core/common/env_var/debug_mode.h"
#include "oneflow/core/boxing/eager_boxing_logger.h"
#include "oneflow/core/boxing/boxing_interpreter_status.h"

namespace oneflow {

namespace {

class NullEagerBoxingLogger final : public EagerBoxingLogger {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NullEagerBoxingLogger);
  NullEagerBoxingLogger() = default;
  ~NullEagerBoxingLogger() override = default;

  void Log(const BoxingInterpreterStatus& status, const std::string& prefix) const override {}
};

class NaiveEagerBoxingLogger final : public EagerBoxingLogger {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NaiveEagerBoxingLogger);
  NaiveEagerBoxingLogger() = default;
  ~NaiveEagerBoxingLogger() override = default;

  void Log(const BoxingInterpreterStatus& status, const std::string& prefix) const override {
    LOG(INFO) << prefix << "Boxing route: " << (status.boxing_routing());
    LOG(INFO) << prefix << "Logical shape: " << (status.logical_shape().ToString());
    LOG(INFO) << prefix << "Altered state of sbp: " << (status.nd_sbp_routing());
    LOG(INFO) << prefix << "Altered state of placement: " << (status.placement_routing());
  }
};

const EagerBoxingLogger* CreateEagerBoxingLogger() {
  if (IsInDebugMode()) {
    return new NaiveEagerBoxingLogger();
  } else {
    return new NullEagerBoxingLogger();
  }
}

}  // namespace

COMMAND(Singleton<const EagerBoxingLogger>::SetAllocated(CreateEagerBoxingLogger()));

}  // namespace oneflow
