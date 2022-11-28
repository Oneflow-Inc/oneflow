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
#ifndef ONEFLOW_USER_KERNELS_COLLECTIVE_COMMUNICATION_INCLUDE_COLLECTIVE_COMMUNICATION_H_
#define ONEFLOW_USER_KERNELS_COLLECTIVE_COMMUNICATION_INCLUDE_COLLECTIVE_COMMUNICATION_H_

#include "oneflow/core/common/auto_registration_factory.h"
#include "oneflow/core/common/switch_func.h"
#include "oneflow/user/kernels/collective_communication/include/communication_context.h"
#include "oneflow/core/ep/include/stream.h"

namespace oneflow {

namespace ccl {

#define REDUCE_TYPE_SEQ      \
  OF_PP_MAKE_TUPLE_SEQ(kSum) \
  OF_PP_MAKE_TUPLE_SEQ(kMax)

enum ReduceType {
  kInvalidReduceFunctorType = 0,
#define DEFINE_REDUCE_TYPE_ENUM_VALUE(enum_value) enum_value,
  OF_PP_FOR_EACH_TUPLE(DEFINE_REDUCE_TYPE_ENUM_VALUE, REDUCE_TYPE_SEQ)
#undef DEFINE_REDUCE_TYPE_ENUM_VALUE
      kReduceTypeSize
};

#define REDUCE_TYPE_CTRV_SEQ      \
  MAKE_TYPED_CTRV_SEQ(ReduceType, \
                      OF_PP_FOR_EACH_TUPLE(OF_PP_I_MAKE_REPLICATE_TUPLE_SEQ, REDUCE_TYPE_SEQ))

class CollectiveCommunication {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CollectiveCommunication);
  CollectiveCommunication() = default;
  virtual ~CollectiveCommunication() = default;
};

template<typename CollectiveCommunicationType, typename... Args>
static std::unique_ptr<CollectiveCommunicationType> NewCollectiveCommunication(
    DeviceType device_type, Args&&... args) {
  std::unique_ptr<CollectiveCommunicationType> collective_communication_entry =
      NewObjUniquePtr<DeviceType, CollectiveCommunicationType>(device_type);
  if (!collective_communication_entry) { return nullptr; }
  collective_communication_entry->Init(std::forward<Args>(args)...);
  return collective_communication_entry;
}

#define REGISTER_COLLECTIVE_COMMUNICATION(device, Base, Derived) \
  REGISTER_CLASS(DeviceType, device, Base, Derived)

}  // namespace ccl

}  // namespace oneflow

#endif  // ONEFLOW_USER_KERNELS_COLLECTIVE_COMMUNICATION_INCLUDE_COLLECTIVE_COMMUNICATION_H_
