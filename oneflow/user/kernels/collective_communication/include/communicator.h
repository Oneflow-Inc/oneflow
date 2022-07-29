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
#ifndef ONEFLOW_CORE_USER_KERNELS_COLLECTIVE_COMMUNICATION_INCLUDE_COMMUNICATOR_H_
#define ONEFLOW_CORE_USER_KERNELS_COLLECTIVE_COMMUNICATION_INCLUDE_COMMUNICATOR_H_

#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/common/auto_registration_factory.h"

namespace oneflow {

namespace ccl {

class Communicator {
 public:
  Communicator() = default;
  virtual ~Communicator() = default;

  virtual void Init(Symbol<ParallelDesc>) = 0;
};

inline std::shared_ptr<Communicator> NewCommunicator(DeviceType device_type,
                                                     Symbol<ParallelDesc> parallel_desc) {
  CHECK_EQ(device_type, parallel_desc->device_type())
      << "device_type not match placement (" << DeviceType_Name(device_type) << " vs. "
      << DeviceType_Name(parallel_desc->device_type()) << ". " << kOfBugIssueUploadPrompt;
  ;
  std::shared_ptr<Communicator> communicator =
      NewObjSharedPtr<DeviceType, Communicator>(device_type);
  communicator->Init(parallel_desc);
  return communicator;
}

inline bool IsCommunicatorRegistered(DeviceType device_type) {
  return IsClassRegistered<DeviceType, Communicator>(device_type);
}

#define REGISTER_COLLECTIVE_COMMUNICATION_COMMUNICATOR(device, Derived) \
  REGISTER_CLASS(DeviceType, device, Communicator, Derived)

}  // namespace ccl

}  // namespace oneflow

#endif  // ONEFLOW_CORE_USER_KERNELS_COLLECTIVE_COMMUNICATION_INCLUDE_COMMUNICATOR_H_
