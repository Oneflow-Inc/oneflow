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
#include "oneflow/core/framework/framework.h"
#include "oneflow/user/summary/events_writer.h"
#include "oneflow/user/summary/env_time.h"
#include "oneflow/user/summary/histogram.h"
#include "oneflow/user/summary/event_writer_helper.h"

#include <time.h>
#include <cstdint>

namespace oneflow {

namespace summary {

template<typename T>
class SummaryWriteScalar final : public user_op::OpKernel {
 public:
  SummaryWriteScalar() = default;
  ~SummaryWriteScalar() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* step = ctx->Tensor4ArgNameAndIndex("step", 0);
    const user_op::Tensor* tag = ctx->Tensor4ArgNameAndIndex("tag", 0);
    const user_op::Tensor* value = ctx->Tensor4ArgNameAndIndex("in", 0);

    T* tvalue = const_cast<T*>(value->dptr<T>());
    CHECK_NOTNULL(tvalue);
    int64_t* istep = const_cast<int64_t*>(step->dptr<int64_t>());
    CHECK_NOTNULL(istep);
    int8_t* ctag = const_cast<int8_t*>(tag->dptr<int8_t>());
    CHECK_NOTNULL(ctag);
    std::string tag_str(reinterpret_cast<char*>(ctag), tag->shape().elem_cnt());
    EventWriterHelper<DeviceType::kCPU, T>::WriteScalarToFile(
        istep[0], static_cast<double>(tvalue[0]), tag_str);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return true; }
};

#define REGISTER_SCALAR_USER_KERNEL(dtype)                \
  REGISTER_USER_KERNEL("summary_write_scalar")            \
      .SetCreateFn<SummaryWriteScalar<dtype>>()           \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "cpu") \
                       & (user_op::HobDataType("in", 0) == GetDataType<dtype>::value));

REGISTER_SCALAR_USER_KERNEL(double)
REGISTER_SCALAR_USER_KERNEL(float)
REGISTER_SCALAR_USER_KERNEL(int64_t)
REGISTER_SCALAR_USER_KERNEL(int32_t)

class CreateSummaryWriter final : public user_op::OpKernel {
 public:
  CreateSummaryWriter() = default;
  ~CreateSummaryWriter() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const std::string& logdir = ctx->Attr<std::string>("logdir");
    Global<EventsWriter>::Get()->Init(logdir);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return true; }
};

REGISTER_USER_KERNEL("create_summary_writer")
    .SetCreateFn<CreateSummaryWriter>()
    .SetIsMatchedHob((user_op::HobDeviceTag() == "cpu"));

class FlushSummaryWriter final : public user_op::OpKernel {
 public:
  FlushSummaryWriter() = default;
  ~FlushSummaryWriter() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    Global<EventsWriter>::Get()->Flush();
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return true; }
};

REGISTER_USER_KERNEL("flush_summary_writer")
    .SetCreateFn<FlushSummaryWriter>()
    .SetIsMatchedHob((user_op::HobDeviceTag() == "cpu"));

template<typename T>
class SummaryWriteHistogram final : public user_op::OpKernel {
 public:
  SummaryWriteHistogram() = default;
  ~SummaryWriteHistogram() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* step = ctx->Tensor4ArgNameAndIndex("step", 0);
    const user_op::Tensor* tag = ctx->Tensor4ArgNameAndIndex("tag", 0);
    const user_op::Tensor* value = ctx->Tensor4ArgNameAndIndex("in", 0);
    int64_t* istep = const_cast<int64_t*>(step->dptr<int64_t>());
    CHECK_NOTNULL(istep);
    int8_t* ctag = const_cast<int8_t*>(tag->dptr<int8_t>());
    CHECK_NOTNULL(ctag);
    std::string tag_str(reinterpret_cast<char*>(ctag), tag->shape().elem_cnt());
    EventWriterHelper<DeviceType::kCPU, T>::WriteHistogramToFile(static_cast<float>(istep[0]),
                                                                 *value, tag_str);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return true; }
};

#define REGISTER_HISTOGRAM_USER_KERNEL(dtype)             \
  REGISTER_USER_KERNEL("summary_write_histogram")         \
      .SetCreateFn<SummaryWriteHistogram<dtype>>()        \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "cpu") \
                       & (user_op::HobDataType("in", 0) == GetDataType<dtype>::value));

REGISTER_HISTOGRAM_USER_KERNEL(double)
REGISTER_HISTOGRAM_USER_KERNEL(float)
REGISTER_HISTOGRAM_USER_KERNEL(int64_t)
REGISTER_HISTOGRAM_USER_KERNEL(int32_t)
REGISTER_HISTOGRAM_USER_KERNEL(int8_t)
REGISTER_HISTOGRAM_USER_KERNEL(uint8_t)

template<typename T>
class SummaryWritePb final : public user_op::OpKernel {
 public:
  SummaryWritePb() = default;
  ~SummaryWritePb() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* step = ctx->Tensor4ArgNameAndIndex("step", 0);
    const user_op::Tensor* value = ctx->Tensor4ArgNameAndIndex("in", 0);
    int64_t* istep = const_cast<int64_t*>(step->dptr<int64_t>());
    CHECK_NOTNULL(istep);
    int8_t* cvalue = const_cast<int8_t*>(value->dptr<int8_t>());
    CHECK_NOTNULL(cvalue);
    std::string value_str(reinterpret_cast<char*>(cvalue), value->shape().elem_cnt());
    EventWriterHelper<DeviceType::kCPU, T>::WritePbToFile(istep[0], value_str);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return true; }
};

REGISTER_USER_KERNEL("summary_write_pb")
    .SetCreateFn<SummaryWritePb<int8_t>>()
    .SetIsMatchedHob((user_op::HobDeviceTag() == "cpu")
                     & (user_op::HobDataType("in", 0) == GetDataType<int8_t>::value));

template<typename T>
class SummaryWriteImage final : public user_op::OpKernel {
 public:
  SummaryWriteImage() = default;
  ~SummaryWriteImage() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* step = ctx->Tensor4ArgNameAndIndex("step", 0);
    const user_op::Tensor* tag = ctx->Tensor4ArgNameAndIndex("tag", 0);
    const user_op::Tensor* value = ctx->Tensor4ArgNameAndIndex("in", 0);
    int64_t* istep = const_cast<int64_t*>(step->dptr<int64_t>());
    CHECK_NOTNULL(istep);
    char* ctag = const_cast<char*>(tag->dptr<char>());
    CHECK_NOTNULL(ctag);
    std::string tag_str(ctag, tag->shape().elem_cnt());
    EventWriterHelper<DeviceType::kCPU, T>::WriteImageToFile(static_cast<int64_t>(istep[0]), *value,
                                                             tag_str);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return true; }
};

REGISTER_USER_KERNEL("summary_write_image")
    .SetCreateFn<SummaryWriteImage<uint8_t>>()
    .SetIsMatchedHob((user_op::HobDeviceTag() == "cpu")
                     & (user_op::HobDataType("in", 0) == GetDataType<uint8_t>::value));

}  // namespace summary

}  // namespace oneflow
