#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/common/device_type.pb.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/customized/utils/event.pb.h"
#include "oneflow/customized/utils/events_writer.h"
#include "oneflow/customized/utils/summary.pb.h"
#include "oneflow/customized/utils/env_time.h"
#include "oneflow/customized/utils/histogram.h"
#include "oneflow/customized/utils/tensor.pb.h"

#include "oneflow/customized/utils/event_write_helper.h"

#include <sys/time.h>
#include <time.h>
#include <cstdint>
#include <memory>

namespace oneflow {

namespace {

template<typename T>
class WriteScalar final : public user_op::OpKernel {
 public:
  WriteScalar() = default;
  ~WriteScalar() = default;

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
    EventWriteHelper<DeviceType::kCPU, T>::WriteScalarToFile(
        istep[0], static_cast<double>(tvalue[0]), reinterpret_cast<char*>(ctag));
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return true; }
};

#define REGISTER_SCALAR_USER_KERNEL(dtype)                            \
  REGISTER_USER_KERNEL("write_scalar")                                \
      .SetCreateFn<WriteScalar<dtype>>()                              \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCPU) \
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
    Global<EventsWriter>::New();
    Global<EventsWriter>::Get()->Init(logdir);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return true; }
};

REGISTER_USER_KERNEL("create_summary_writer")
    .SetCreateFn<CreateSummaryWriter>()
    .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCPU));

class FlushEventWriter final : public user_op::OpKernel {
 public:
  FlushEventWriter() = default;
  ~FlushEventWriter() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    Global<EventsWriter>::Get()->Flush();
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return true; }
};

REGISTER_USER_KERNEL("flush_event_writer")
    .SetCreateFn<FlushEventWriter>()
    .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCPU));

template<typename T>
class WriteHistogram final : public user_op::OpKernel {
 public:
  WriteHistogram() = default;
  ~WriteHistogram() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* step = ctx->Tensor4ArgNameAndIndex("step", 0);
    const user_op::Tensor* tag = ctx->Tensor4ArgNameAndIndex("tag", 0);
    const user_op::Tensor* value = ctx->Tensor4ArgNameAndIndex("in", 0);
    int64_t* istep = const_cast<int64_t*>(step->dptr<int64_t>());
    CHECK_NOTNULL(istep);
    int8_t* ctag = const_cast<int8_t*>(tag->dptr<int8_t>());
    CHECK_NOTNULL(ctag);
    EventWriteHelper<DeviceType::kCPU, T>::WriteHistogramToFile(
        static_cast<float>(istep[0]), *value, reinterpret_cast<char*>(ctag));
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return true; }
};

#define REGISTER_HISTOGRAM_USER_KERNEL(dtype)                         \
  REGISTER_USER_KERNEL("write_histogram")                             \
      .SetCreateFn<WriteHistogram<dtype>>()                           \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCPU) \
                       & (user_op::HobDataType("in", 0) == GetDataType<dtype>::value));

REGISTER_HISTOGRAM_USER_KERNEL(double)
REGISTER_HISTOGRAM_USER_KERNEL(float)
REGISTER_HISTOGRAM_USER_KERNEL(int64_t)
REGISTER_HISTOGRAM_USER_KERNEL(int32_t)
REGISTER_HISTOGRAM_USER_KERNEL(int8_t)
REGISTER_HISTOGRAM_USER_KERNEL(uint8_t)

template<typename T>
class WritePb final : public user_op::OpKernel {
 public:
  WritePb() = default;
  ~WritePb() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* step = ctx->Tensor4ArgNameAndIndex("step", 0);
    const user_op::Tensor* value = ctx->Tensor4ArgNameAndIndex("in", 0);
    int64_t* istep = const_cast<int64_t*>(step->dptr<int64_t>());
    CHECK_NOTNULL(istep);
    int8_t* cvalue = const_cast<int8_t*>(value->dptr<int8_t>());
    CHECK_NOTNULL(value);
    EventWriteHelper<DeviceType::kCPU, T>::WritePbToFile(istep[0], reinterpret_cast<char*>(cvalue));
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return true; }
};

REGISTER_USER_KERNEL("write_pb")
    .SetCreateFn<WritePb<int8_t>>()
    .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCPU)
                     & (user_op::HobDataType("in", 0) == GetDataType<int8_t>::value));

template<typename T>
class WriteImage final : public user_op::OpKernel {
 public:
  WriteImage() = default;
  ~WriteImage() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* step = ctx->Tensor4ArgNameAndIndex("step", 0);
    const user_op::Tensor* tag = ctx->Tensor4ArgNameAndIndex("tag", 0);
    const user_op::Tensor* value = ctx->Tensor4ArgNameAndIndex("in", 0);
    int64_t* istep = const_cast<int64_t*>(step->dptr<int64_t>());
    CHECK_NOTNULL(istep);
    char* ctag = const_cast<char*>(tag->dptr<char>());
    CHECK_NOTNULL(ctag);
    EventWriteHelper<DeviceType::kCPU, T>::WriteImageToFile(static_cast<int64_t>(istep[0]), value,
                                                            ctag);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return true; }
};

REGISTER_USER_KERNEL("write_image")
    .SetCreateFn<WriteImage<uint8_t>>()
    .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCPU)
                     & (user_op::HobDataType("in", 0) == GetDataType<uint8_t>::value));

}  // namespace
}  // namespace oneflow