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
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/common/blocking_counter.h"
#include "oneflow/core/common/tensor_buffer.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/thread/thread_manager.h"
#include "oneflow/user/image/random_crop_generator.h"
#include "oneflow/user/image/image_util.h"
#include "oneflow/user/kernels/random_crop_kernel_state.h"
#include "oneflow/user/kernels/op_kernel_wrapper.h"
#include "oneflow/user/kernels/random_seed_util.h"
#include "oneflow/user/image/jpeg_decoder.h"

#include <opencv2/opencv.hpp>
#include <jpeglib.h>

namespace oneflow {

namespace {

template<typename T>
void DecodeOneRawOFRecord(const Feature& feature, T* dptr, int64_t sample_elem_cnt, bool truncate,
                          bool dim1_varying_length) {
  if (feature.has_bytes_list()) {
    CHECK_EQ(feature.bytes_list().value_size(), 1);
    const auto& value0 = feature.bytes_list().value(0);
    auto in_dptr = reinterpret_cast<const int8_t*>(value0.c_str());
    sample_elem_cnt = std::min<int64_t>(sample_elem_cnt, value0.size());
    std::transform(in_dptr, in_dptr + sample_elem_cnt, dptr,
                   [](int8_t v) { return static_cast<T>(v); });
  }
#define DEFINE_ONE_ELIF(PbT, CppT)                                                       \
  else if (feature.has_##PbT##_list()) {                                                 \
    const auto& list = feature.PbT##_list();                                             \
    const CppT* in_dptr = list.value().data();                                           \
    const int64_t padding_elem_num = truncate ? sample_elem_cnt - list.value_size() : 0; \
    if (truncate) {                                                                      \
      sample_elem_cnt = std::min<int64_t>(sample_elem_cnt, list.value_size());           \
    } else {                                                                             \
      if (dim1_varying_length) {                                                         \
        sample_elem_cnt = list.value_size();                                             \
      } else {                                                                           \
        CHECK_EQ(sample_elem_cnt, list.value_size());                                    \
      }                                                                                  \
    }                                                                                    \
    std::transform(in_dptr, in_dptr + sample_elem_cnt, dptr,                             \
                   [](CppT v) { return static_cast<T>(v); });                            \
    if (padding_elem_num > 0) {                                                          \
      std::memset(dptr + sample_elem_cnt, 0, padding_elem_num * sizeof(T));              \
    }                                                                                    \
  }
  DEFINE_ONE_ELIF(float, float)
  DEFINE_ONE_ELIF(double, double)
  DEFINE_ONE_ELIF(int32, int32_t)
  DEFINE_ONE_ELIF(int64, int64_t)
#undef DEFINE_ONE_ELIF
  else {
    UNIMPLEMENTED();
  }
}

}  // namespace

template<typename T>
class OFRecordRawDecoderKernel final : public user_op::OpKernel {
 public:
  OFRecordRawDecoderKernel() = default;
  ~OFRecordRawDecoderKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    user_op::Tensor* in_blob = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out_blob = ctx->Tensor4ArgNameAndIndex("out", 0);
    // TODO(chengcheng): remove record num in record blob, fix by shape elem cnt
    int64_t record_num = in_blob->shape_view().At(0);
    int64_t sample_elem_cnt = out_blob->shape_view().Count(1);
    CHECK(record_num > 0);
    const OFRecord* records = in_blob->dptr<OFRecord>();
    T* out_dptr = out_blob->mut_dptr<T>();
    const std::string& name = ctx->Attr<std::string>("name");

    bool truncate = ctx->Attr<bool>("truncate");
    bool dim1_varying_length = ctx->Attr<bool>("dim1_varying_length");

    MultiThreadLoop(record_num, [&](size_t i) {
      const OFRecord& record = *(records + i);
      T* dptr = out_dptr + i * sample_elem_cnt;
      CHECK(record.feature().find(name) != record.feature().end())
          << "Field " << name << " not found";
      const Feature& feature = record.feature().at(name);
      DecodeOneRawOFRecord(feature, dptr, sample_elem_cnt, truncate, dim1_varying_length);
    });
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_RAW_DECODER_KERNEL(dtype)                                       \
  REGISTER_USER_KERNEL("ofrecord_raw_decoder")                                   \
      .SetCreateFn<OFRecordRawDecoderKernel<dtype>>()                            \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCPU)            \
                       && (user_op::HobDataType("in", 0) == DataType::kOFRecord) \
                       && (user_op::HobDataType("out", 0) == GetDataType<dtype>::value));

REGISTER_RAW_DECODER_KERNEL(char)
REGISTER_RAW_DECODER_KERNEL(float)
REGISTER_RAW_DECODER_KERNEL(double)
REGISTER_RAW_DECODER_KERNEL(int8_t)
REGISTER_RAW_DECODER_KERNEL(int32_t)
REGISTER_RAW_DECODER_KERNEL(int64_t)
REGISTER_RAW_DECODER_KERNEL(uint8_t)

class OFRecordBytesDecoderKernel final : public user_op::OpKernel {
 public:
  OFRecordBytesDecoderKernel() = default;
  ~OFRecordBytesDecoderKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    CHECK_EQ(out->shape_view(), in->shape_view());
    CHECK_EQ(in->data_type(), DataType::kOFRecord);
    CHECK_EQ(out->data_type(), DataType::kTensorBuffer);
    const int64_t num_instances = in->shape_view().elem_cnt();
    const auto* records = in->dptr<OFRecord>();
    auto* buffers = out->mut_dptr<TensorBuffer>();
    const std::string& name = ctx->Attr<std::string>("name");
    MultiThreadLoop(num_instances, [&](size_t i) {
      const OFRecord& record = *(records + i);
      TensorBuffer* buffer = buffers + i;
      auto it = record.feature().find(name);
      CHECK(it != record.feature().end()) << "Field " << name << " not found";
      const Feature& feature = it->second;
      CHECK(feature.has_bytes_list());
      CHECK_EQ(feature.bytes_list().value_size(), 1);
      const int64_t size = feature.bytes_list().value(0).size();
      buffer->Resize(Shape({size}), DataType::kUInt8);
      memcpy(buffer->mut_data(), feature.bytes_list().value(0).data(), size);
    });
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("ofrecord_bytes_decoder")
    .SetCreateFn<OFRecordBytesDecoderKernel>()
    .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCPU)
                     && (user_op::HobDataType("in", 0) == DataType::kOFRecord)
                     && (user_op::HobDataType("out", 0) == DataType::kTensorBuffer));

namespace {

void DecodeRandomCropImageFromOneRecord(const OFRecord& record, TensorBuffer* buffer,
                                        const std::string& name, const std::string& color_space,
                                        RandomCropGenerator* random_crop_gen) {
  CHECK(record.feature().find(name) != record.feature().end()) << "Field " << name << " not found";
  const Feature& feature = record.feature().at(name);
  CHECK(feature.has_bytes_list());
  CHECK(feature.bytes_list().value_size() == 1);
  const std::string& src_data = feature.bytes_list().value(0);
  cv::Mat image;

  if (JpegPartialDecodeRandomCropImage(reinterpret_cast<const unsigned char*>(src_data.data()),
                                       src_data.size(), random_crop_gen, nullptr, 0, &image)) {
    // convert color space
    // jpeg decode output RGB
    if (ImageUtil::IsColor(color_space) && color_space != "RGB") {
      ImageUtil::ConvertColor("RGB", image, color_space, image);
    }
  } else {
    OpenCvPartialDecodeRandomCropImage(reinterpret_cast<const unsigned char*>(src_data.data()),
                                       src_data.size(), random_crop_gen, color_space, image);
    // convert color space
    // opencv decode output BGR
    if (ImageUtil::IsColor(color_space) && color_space != "BGR") {
      ImageUtil::ConvertColor("BGR", image, color_space, image);
    }
  }

  int W = image.cols;
  int H = image.rows;

  CHECK(image.isContinuous());
  const int c = ImageUtil::IsColor(color_space) ? 3 : 1;
  CHECK_EQ(c, image.channels());
  Shape image_shape({H, W, c});
  buffer->Resize(image_shape, DataType::kUInt8);
  CHECK_EQ(image_shape.elem_cnt(), buffer->nbytes());
  CHECK_EQ(image_shape.elem_cnt(), image.total() * image.elemSize());
  memcpy(buffer->mut_data<uint8_t>(), image.ptr(), image_shape.elem_cnt());
}

}  // namespace

class OFRecordImageDecoderRandomCropKernel final : public user_op::OpKernel {
 public:
  OFRecordImageDecoderRandomCropKernel() = default;
  ~OFRecordImageDecoderRandomCropKernel() override = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return CreateRandomCropKernelState(ctx);
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state,
               const user_op::OpKernelCache*) const override {
    auto* crop_window_generators = dynamic_cast<RandomCropKernelState*>(state);
    CHECK_NOTNULL(crop_window_generators);
    user_op::Tensor* out_blob = ctx->Tensor4ArgNameAndIndex("out", 0);
    int64_t record_num = out_blob->shape_view().At(0);
    CHECK(record_num > 0);
    user_op::Tensor* in_blob = ctx->Tensor4ArgNameAndIndex("in", 0);
    CHECK_EQ(out_blob->shape_view(), in_blob->shape_view());
    const OFRecord* records = in_blob->dptr<OFRecord>();
    TensorBuffer* buffers = out_blob->mut_dptr<TensorBuffer>();
    const std::string& name = ctx->Attr<std::string>("name");
    const std::string& color_space = ctx->Attr<std::string>("color_space");

    MultiThreadLoop(record_num, [&](size_t i) {
      const OFRecord& record = *(records + i);
      TensorBuffer* buffer = buffers + i;
      RandomCropGenerator* gen = crop_window_generators->GetGenerator(i);
      DecodeRandomCropImageFromOneRecord(record, buffer, name, color_space, gen);
    });
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("ofrecord_image_decoder_random_crop")
    .SetCreateFn<OFRecordImageDecoderRandomCropKernel>()
    .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCPU)
                     && (user_op::HobDataType("in", 0) == DataType::kOFRecord)
                     && (user_op::HobDataType("out", 0) == DataType::kTensorBuffer));

class OFRecordImageDecoderKernel final : public user_op::OpKernel {
 public:
  OFRecordImageDecoderKernel() = default;
  ~OFRecordImageDecoderKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    user_op::Tensor* out_blob = ctx->Tensor4ArgNameAndIndex("out", 0);
    int64_t record_num = out_blob->shape_view().At(0);
    CHECK(record_num > 0);
    user_op::Tensor* in_blob = ctx->Tensor4ArgNameAndIndex("in", 0);
    CHECK_EQ(out_blob->shape_view(), in_blob->shape_view());
    const OFRecord* records = in_blob->dptr<OFRecord>();
    TensorBuffer* buffers = out_blob->mut_dptr<TensorBuffer>();
    const std::string& name = ctx->Attr<std::string>("name");
    const std::string& color_space = ctx->Attr<std::string>("color_space");

    MultiThreadLoop(record_num, [&](size_t i) {
      const OFRecord& record = *(records + i);
      TensorBuffer* buffer = buffers + i;
      DecodeRandomCropImageFromOneRecord(record, buffer, name, color_space, nullptr);
    });
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("ofrecord_image_decoder")
    .SetCreateFn<OFRecordImageDecoderKernel>()
    .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCPU)
                     && (user_op::HobDataType("in", 0) == DataType::kOFRecord)
                     && (user_op::HobDataType("out", 0) == DataType::kTensorBuffer));

}  // namespace oneflow
