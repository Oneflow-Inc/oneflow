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
#include "oneflow/core/comm_network/ibverbs/ibverbs_memory_desc.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/platform/include/ibv.h"

#if defined(WITH_RDMA) && defined(OF_PLATFORM_POSIX)

namespace oneflow {

IBVerbsMemDesc::IBVerbsMemDesc(ibv_pd* pd, void* mem_ptr, size_t byte_size)
    : mem_ptr_(mem_ptr), mem_size_(byte_size) {
  mr_ = ibv::wrapper.ibv_reg_mr_wrap(
      pd, mem_ptr, byte_size,
      IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);
  PCHECK(mr_);
}

IBVerbsMemDesc::~IBVerbsMemDesc() { PCHECK(ibv::wrapper.ibv_dereg_mr(mr_) == 0); }

}  // namespace oneflow

#endif  // WITH_RDMA && OF_PLATFORM_POSIX
