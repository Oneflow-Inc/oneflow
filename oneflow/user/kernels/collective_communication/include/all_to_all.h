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
#ifndef ONEFLOW_USER_KERNELS_COLLECTIVE_COMMUNICATION_INCLUDE_ALL_TO_ALL_H_
#define ONEFLOW_USER_KERNELS_COLLECTIVE_COMMUNICATION_INCLUDE_ALL_TO_ALL_H_

#include "oneflow/user/kernels/collective_communication/include/collective_communication.h"

namespace oneflow {

namespace ccl {

class AllToAll : public CollectiveCommunication {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AllToAll);
  AllToAll() = default;
  ~AllToAll() override = default;

  virtual void Init(DataType send_dtype, DataType recv_dtype, size_t rank_count) = 0;

  // for normal alltoall（balanced send/resv count)
  virtual void Launch(ep::Stream* stream, void* send, int64_t send_count, void* recv,
                      int64_t recv_count, ccl::CclComm ccl_comm) const = 0;

  // for unbalanced all to all(e.g. nccl all2all using send/recv; hccl HcclAlltoAllV)
  virtual void Launch(ep::Stream* stream, void* send, const void* send_counts,
                      const void* send_offsets, void* recv, const void* recv_counts,
                      const void* recv_offsets, ccl::CclComm ccl_comm) const = 0;
};

inline bool IsAllToAllRegistered(DeviceType device_type) {
  return IsClassRegistered<DeviceType, AllToAll>(device_type);
}

}  // namespace ccl

}  // namespace oneflow

#endif  // ONEFLOW_USER_KERNELS_COLLECTIVE_COMMUNICATION_INCLUDE_ALL_TO_ALL_H_
