#ifndef ONEFLOW_CORE_VM_STREAM_GET_STREAM_TYPE_H_
#define ONEFLOW_CORE_VM_STREAM_GET_STREAM_TYPE_H_

#include "oneflow/core/framework/stream_role.h"
#include "oneflow/core/common/singleton_ptr.h"
#include "oneflow/core/vm/async_cuda_stream_type.h"
#include "oneflow/core/vm/control_stream_type.h"
#include "oneflow/core/vm/cpu_stream_type.h"
#include "oneflow/core/vm/critical_section_stream_type.h"
#include "oneflow/core/vm/cuda_copy_d2h_stream_type.h"
#include "oneflow/core/vm/cuda_copy_h2d_stream_type.h"
#include "oneflow/core/vm/cuda_stream_type.h"
#include "oneflow/core/vm/lazy_job_stream_type.h"
#include "oneflow/core/vm/stream_get_stream_type.h"

namespace oneflow {

struct GetStreamType {
  static Maybe<const vm::InstructionType*> Case(StreamRoleCase<StreamRole::kInvalid>,
                                                DeviceType device_type) {  // NOLINT
    UNIMPLEMENTED_THEN_RETURN();
  }
  static Maybe<const vm::InstructionType*> Case(StreamRoleCase<StreamRole::kCompute>,
                                        DeviceType device_type) {
    if (device_type == DeviceType::kCPU) {
      return SingletonPtr<vm::CpuStreamType>();
    } else if (device_type == DeviceType::kCUDA) {
      return SingletonPtr<vm::CudaStreamType>();
    } else {
      UNIMPLEMENTED_THEN_RETURN();
    }
  }
  static Maybe<const vm::InstructionType*> Case(StreamRoleCase<StreamRole::kHost2Device>,
                                        DeviceType device_type) {
    if (device_type == DeviceType::kCUDA) {
      return SingletonPtr<vm::CudaCopyH2DStreamType>();
    } else {
      UNIMPLEMENTED_THEN_RETURN();
    }
  }
  static Maybe<const std::string&> Case(StreamRoleCase<StreamRole::kDevice2Host>,
                                        DeviceType device_type) {
    if (device_type == DeviceType::kCUDA) {
      return SingletonPtr<vm::CudaCopyD2HStreamType>();
    } else {
      UNIMPLEMENTED_THEN_RETURN();
    }
  }
  static Maybe<const std::string&> Case(StreamRoleCase<StreamRole::kSyncedLaunchedCommNet>,
                                        DeviceType device_type) {
    if (device_type == DeviceType::kCPU) {
      return SingletonPtr<vm::CpuStreamType>();
    } else if (device_type == DeviceType::kCUDA) {
      return SingletonPtr<vm::CudaStreamType>();
    } else {
      UNIMPLEMENTED_THEN_RETURN();
    }
  }
  static Maybe<const std::string&> Case(StreamRoleCase<StreamRole::kAsyncedLaunchedCommNet>,
                                        DeviceType device_type) {
    if (device_type == DeviceType::kCPU) {
      return SingletonPtr<vm::CpuStreamType>();
    } else if (device_type == DeviceType::kCUDA) {
      return SingletonPtr<vm::AsyncCudaStreamType>();
    } else {
      UNIMPLEMENTED_THEN_RETURN();
    }
  }
  static Maybe<const std::string&> Case(StreamRoleCase<StreamRole::kBarrier>,
                                        DeviceType device_type) {
    return SingletonPtr<vm::ControlStreamType>();
  }
  static Maybe<const std::string&> Case(StreamRoleCase<StreamRole::kCriticalSection>,
                                        DeviceType device_type) {
    return SingletonPtr<vm::CriticalSectionStreamType>();
  }
  static Maybe<const std::string&> Case(StreamRoleCase<StreamRole::kLazyJobLauncher>,
                                        DeviceType device_type) {
    return SingletonPtr<vm::LazyJobStreamType>();
  }
};

}

#endif  // ONEFLOW_CORE_VM_STREAM_GET_STREAM_TYPE_H_
