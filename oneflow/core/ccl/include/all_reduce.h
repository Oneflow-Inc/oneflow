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
#ifndef ONEFLOW_CORE_CCL_INCLUDE_ALL_REDUCE_H_
#define ONEFLOW_CORE_CCL_INCLUDE_ALL_REDUCE_H_

#include "oneflow/core/ccl/include/collective_communication.h"

namespace oneflow {

namespace ccl {

namespace collective_communication {

class AllReduce : public CollectiveCommunication {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AllReduce);
  AllReduce() = default;
  ~AllReduce() override = default;

  virtual void Launch(ep::Stream* stream, const void* in, void* out, size_t elem_cnt,
                      const std::shared_ptr<Communicator>& communicator) const = 0;
};

class AllReduceFactory : public CollectiveCommunicationFactory<AllReduce> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AllReduceFactory);
  AllReduceFactory() = default;
  ~AllReduceFactory() override = default;

  virtual std::unique_ptr<AllReduce> New(DataType dtype, ReduceType reduce_type) = 0;
};

inline bool IsAllReduceRegistered(DeviceType device_type) {
  return IsClassRegistered<DeviceType, AllReduceFactory>(device_type);
}

}  // namespace collective_communication

}  // namespace ccl

}  // namespace oneflow

#endif  // ONEFLOW_CORE_CCL_INCLUDE_ALL_REDUCE_H_
