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
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/ccl/ccl.h"
#include "oneflow/core/job/rank_group.h"
#include "oneflow/core/framework/transport_util.h"
#include "oneflow/user/kernels/collective_communication/include/recv.h"

namespace oneflow {

namespace ccl {

// Use CpuRecvImpl to avoid name conflict
class CpuRecvImpl final : public Recv {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CpuRecvImpl);
  CpuRecvImpl() : size_of_dtype_(0) {}
  ~CpuRecvImpl() = default;

  void Init(DataType datatype) override {
    CHECK(IsTriviallyCopyableDataType(datatype));
    this->size_of_dtype_ = GetSizeOfDataType(datatype);
  }

  void Launch(ep::Stream* stream, void* out, size_t elem_cnt, int64_t src) const override {
    size_t buffer_size = elem_cnt * size_of_dtype_;
    CHECK_JUST(CpuRecv(out, buffer_size, src));
  }

 private:
  size_t size_of_dtype_;
};

REGISTER_COLLECTIVE_COMMUNICATION(DeviceType::kCPU, Recv, CpuRecvImpl);

}  // namespace ccl

}  // namespace oneflow
