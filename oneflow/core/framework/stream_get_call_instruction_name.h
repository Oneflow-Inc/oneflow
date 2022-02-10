#ifndef ONEFLOW_CORE_FRAMEWORK_STREAM_GET_CALL_INSTRUCTION_NAME_H_
#define ONEFLOW_CORE_FRAMEWORK_STREAM_GET_CALL_INSTRUCTION_NAME_H_

#include <glog/logging.h>
#include <string>
#include "oneflow/core/common/device.h"
#include "oneflow/core/common/stream_role.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/framework/to_string.h"

namespace oneflow {

struct GetCallInstructionName {
  static Maybe<std::string> Case(SR<StreamRole::kInvalid>, DeviceType device_type) { // NOLINT
    UNIMPLEMENTED_THEN_RETURN();
  }
  static Maybe<std::string> Case(SR<StreamRole::kCompute>, DeviceType device_type) {
    return *JUST(DeviceTag4DeviceType(device_type)) + ".LocalCallOpKernel";
  }
  static Maybe<std::string> Case(SR<StreamRole::kHost2Device>, DeviceType device_type) {
    CHECK_EQ_OR_RETURN(device_type, kCUDA);
    return "cuda_h2d.LocalCallOpKernel";
  }
  static Maybe<std::string> Case(SR<StreamRole::kDevice2Host>, DeviceType device_type) {
    CHECK_EQ_OR_RETURN(device_type, kCUDA);
    return "cuda_d2h.LocalCallOpKernel";
  }
  static Maybe<std::string> Case(SR<StreamRole::kSyncedLaunchedCC>, DeviceType device_type) {
    if (device_type == kCPU) { return "cpu.LocalCallOpKernel"; }
    CHECK_EQ_OR_RETURN(device_type, kCUDA);
    return "gpu.LocalCallOpKernel";
  }
  static Maybe<std::string> Case(SR<StreamRole::kAsyncedLaunchedCC>, DeviceType device_type) {
    if (device_type == kCPU) { return "cpu.LocalCallOpKernel"; }
    CHECK_EQ_OR_RETURN(device_type, kCUDA);
    return "async.gpu.LocalCallOpKernel";
  }
  static Maybe<std::string> Case(SR<StreamRole::kCriticalSection>, DeviceType device_type) {
    UNIMPLEMENTED_THEN_RETURN();
  }
};

}

#endif  // ONEFLOW_CORE_FRAMEWORK_STREAM_GET_CALL_INSTRUCTION_NAME_H_
