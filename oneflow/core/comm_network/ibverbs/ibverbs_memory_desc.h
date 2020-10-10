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
#ifndef ONEFLOW_CORE_COMM_NETWORK_IBVERBS_IBVERBS_MEMORY_DESC_H_
#define ONEFLOW_CORE_COMM_NETWORK_IBVERBS_IBVERBS_MEMORY_DESC_H_

#include "oneflow/core/common/platform.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/comm_network/ibverbs/ibverbs.pb.h"

#if defined(WITH_RDMA) && defined(OF_PLATFORM_POSIX)

#include <infiniband/verbs.h>

namespace oneflow {

class IBVerbsMemDesc final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(IBVerbsMemDesc);
  IBVerbsMemDesc() = delete;
  IBVerbsMemDesc(ibv_pd* pd, void* mem_ptr, size_t byte_size);
  ~IBVerbsMemDesc();

  const std::vector<ibv_sge>& sge_vec() const { return sge_vec_; }

  IBVerbsMemDescProto ToProto();

 private:
  std::vector<ibv_sge> sge_vec_;
  std::vector<ibv_mr*> mr_vec_;
};

}  // namespace oneflow

#endif  // WITH_RDMA && OF_PLATFORM_POSIX

#endif  // ONEFLOW_CORE_COMM_NETWORK_IBVERBS_IBVERBS_MEMORY_DESC_H_
