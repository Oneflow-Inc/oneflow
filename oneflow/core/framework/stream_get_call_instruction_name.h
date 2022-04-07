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
#ifndef ONEFLOW_CORE_FRAMEWORK_STREAM_GET_CALL_INSTRUCTION_NAME_H_
#define ONEFLOW_CORE_FRAMEWORK_STREAM_GET_CALL_INSTRUCTION_NAME_H_

#include <glog/logging.h>
#include <string>
#include "oneflow/core/common/decorator.h"
#include "oneflow/core/common/stream_role.h"
#include "oneflow/core/common/device_type.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/framework/device.h"
#include "oneflow/core/framework/to_string.h"

namespace oneflow {

struct GetCallInstructionName {
  static std::string Case(StreamRoleCase<StreamRole::kInvalid>,
                                        Symbol<Device> device) {  // NOLINT
    UNIMPLEMENTED();
  }
  static std::string Case(StreamRoleCase<StreamRole::kCompute>,
                                        Symbol<Device> device) {
    static constexpr auto* Get = DECORATE(&Call::Compute, ThreadLocal);
    return Get(device);
  }
  static std::string Case(StreamRoleCase<StreamRole::kHost2Device>,
                                        Symbol<Device> device) {
    static constexpr auto* Get = DECORATE(&Call::Host2Device, ThreadLocal);
    return Get(device);
  }
  static std::string Case(StreamRoleCase<StreamRole::kDevice2Host>,
                                        Symbol<Device> device) {
    static constexpr auto* Get = DECORATE(&Call::Device2Host, ThreadLocal);
    return Get(device);
  }
  static std::string Case(StreamRoleCase<StreamRole::kSyncedLaunchedCommNet>,
                                        Symbol<Device> device) {
    static constexpr auto* Get = DECORATE(&Call::SyncedLaunchedCommNet, ThreadLocal);
    return Get(device);
  }
  static std::string Case(StreamRoleCase<StreamRole::kAsyncedLaunchedCommNet>,
                                        Symbol<Device> device) {
    static constexpr auto* Get = DECORATE(&Call::AsyncedLaunchedCommNet, ThreadLocal);
    return Get(device);
  }
  static std::string Case(StreamRoleCase<StreamRole::kCriticalSection>,
                                        Symbol<Device> device) {
    static constexpr auto* Get = DECORATE(&Call::CriticalSection, ThreadLocal);
    return Get(device);
  }

 private:
  struct Call {
    static std::string Invalid(Symbol<Device> device) {  // NOLINT
      UNIMPLEMENTED();
    }
    static std::string Compute(Symbol<Device> device) {
      return device->type() + ".LocalCallOpKernel";
    }
    static std::string Host2Device(Symbol<Device> device) {
      CHECK_EQ_OR_RETURN(device->enum_type(), kCUDA);
      return std::string("cuda_h2d.LocalCallOpKernel");
    }
    static std::string Device2Host(Symbol<Device> device) {
      CHECK_EQ_OR_RETURN(device->enum_type(), kCUDA);
      return std::string("cuda_d2h.LocalCallOpKernel");
    }
    static std::string SyncedLaunchedCommNet(Symbol<Device> device) {
      if (device->enum_type() == kCPU) { return std::string("cpu.LocalCallOpKernel"); }
      CHECK_EQ_OR_RETURN(device->enum_type(), kCUDA);
      return std::string("gpu.LocalCallOpKernel");
    }
    static std::string AsyncedLaunchedCommNet(Symbol<Device> device) {
      if (device->enum_type() == kCPU) { return std::string("cpu.LocalCallOpKernel"); }
      CHECK_EQ_OR_RETURN(device->enum_type(), kCUDA);
      return std::string("async.gpu.LocalCallOpKernel");
    }
    static std::string CriticalSection(Symbol<Device> device) {
      UNIMPLEMENTED_THEN_RETURN();
    }
  };
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_STREAM_GET_CALL_INSTRUCTION_NAME_H_
