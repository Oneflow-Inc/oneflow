#include "oneflow/core/framework/framework.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/common/blocking_counter.h"
#include "oneflow/core/common/tensor_buffer.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/thread/thread_manager.h"
#include "oneflow/customized/image/random_crop_generator.h"
#include "oneflow/customized/image/image_util.h"
#include "oneflow/customized/kernels/op_kernel_state_wrapper.h"
#include "oneflow/customized/kernels/random_seed_util.h"

#include <opencv2/opencv.hpp>

namespace oneflow {

namespace {

template<typename T>
void DecodeOneRawOFRecord(const Feature& feature, T* dptr, int64_t sample_elem_cnt,
                          bool dim1_varying_length, bool auto_zero_padding) {
  if (feature.has_bytes_list()) {
    CHECK_EQ(feature.bytes_list().value_size(), 1);
    const auto& value0 = feature.bytes_list().value(0);
    auto in_dptr = reinterpret_cast<const int8_t*>(value0.c_str());
    sample_elem_cnt = std::min<int64_t>(sample_elem_cnt, value0.size());
    CopyElem<int8_t, T>(in_dptr, dptr, sample_elem_cnt);
  }
#define DEFINE_ONE_ELIF(PbT, CppT)                                                                \
  else if (feature.has_##PbT##_list()) {                                                          \
    const auto& list = feature.PbT##_list();                                                      \
    const CppT* in_dptr = list.value().data();                                                    \
    const int64_t padding_elem_num = auto_zero_padding ? sample_elem_cnt - list.value_size() : 0; \
    if (dim1_varying_length || auto_zero_padding) {                                               \
      CHECK_LE(list.value_size(), sample_elem_cnt);                                               \
      sample_elem_cnt = list.value_size();                                                        \
    } else {                                                                                      \
      CHECK_EQ(sample_elem_cnt, list.value_size());                                               \
    }                                                                                             \
    CopyElem<CppT, T>(in_dptr, dptr, sample_elem_cnt);                                            \
    if (padding_elem_num > 0) {                                                                   \
      std::memset(dptr + sample_elem_cnt, 0, padding_elem_num * sizeof(T));                       \
    }                                                                                             \
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
    int64_t record_num = in_blob->shape().At(0);
    int64_t sample_elem_cnt = out_blob->shape().Count(1);
    CHECK(record_num > 0);
    OFRecord* records = in_blob->mut_dptr<OFRecord>();
    T* out_dptr = out_blob->mut_dptr<T>();
    const std::string& name = ctx->Attr<std::string>("name");

    bool auto_zero_padding = ctx->Attr<bool>("auto_zero_padding");
    bool dim1_varying_length = ctx->Attr<bool>("dim1_varying_length");

    MultiThreadLoop(record_num, [&](size_t i) {
      const OFRecord& record = *(records + i);
      T* dptr = out_dptr + i * sample_elem_cnt;
      CHECK(record.feature().find(name) != record.feature().end())
          << "Field " << name << " not found";
      const Feature& feature = record.feature().at(name);
      DecodeOneRawOFRecord(feature, dptr, sample_elem_cnt, auto_zero_padding, dim1_varying_length);
    });
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_RAW_DECODER_KERNEL(dtype)                                    \
  REGISTER_USER_KERNEL("ofrecord_raw_decoder")                                \
      .SetCreateFn<OFRecordRawDecoderKernel<dtype>>()                         \
      .SetIsMatchedHob(user_op::HobDeviceType() == DeviceType::kCPU           \
                       & user_op::HobDataType("in", 0) == DataType::kOFRecord \
                       & user_op::HobDataType("out", 0) == GetDataType<dtype>::value);

REGISTER_RAW_DECODER_KERNEL(char)
REGISTER_RAW_DECODER_KERNEL(float)
REGISTER_RAW_DECODER_KERNEL(double)
REGISTER_RAW_DECODER_KERNEL(int8_t)
REGISTER_RAW_DECODER_KERNEL(int32_t)
REGISTER_RAW_DECODER_KERNEL(int64_t)
REGISTER_RAW_DECODER_KERNEL(uint8_t)

namespace {

void DecodeRandomCropImageFromOneRecord(const OFRecord& record, TensorBuffer* buffer,
                                        const std::string& name, const std::string& color_space,
                                        RandomCropGenerator* random_crop_gen) {
  CHECK(record.feature().find(name) != record.feature().end()) << "Field " << name << " not found";
  const Feature& feature = record.feature().at(name);
  CHECK(feature.has_bytes_list());
  CHECK(feature.bytes_list().value_size() == 1);
  const std::string& src_data = feature.bytes_list().value(0);

  // cv::_InputArray image_data(src_data.data(), src_data.size());
  // cv::Mat image = cv::imdecode(image_data, cv::IMREAD_ANYCOLOR);
  cv::Mat image =
      cv::imdecode(cv::Mat(1, src_data.size(), CV_8UC1, (void*)(src_data.data())),  // NOLINT
                   ImageUtil::IsColor(color_space) ? cv::IMREAD_COLOR : cv::IMREAD_GRAYSCALE);
  int W = image.cols;
  int H = image.rows;

  // random crop
  if (random_crop_gen != nullptr) {
    CHECK(image.data != nullptr);
    cv::Mat image_roi;
    CropWindow crop;
    random_crop_gen->GenerateCropWindow({H, W}, &crop);
    const int y = crop.anchor.At(0);
    const int x = crop.anchor.At(1);
    const int newH = crop.shape.At(0);
    const int newW = crop.shape.At(1);
    CHECK(newW > 0 && newW <= W);
    CHECK(newH > 0 && newH <= H);
    cv::Rect roi(x, y, newW, newH);
    image(roi).copyTo(image_roi);
    image = image_roi;
    W = image.cols;
    H = image.rows;
    CHECK(W == newW);
    CHECK(H == newH);
  }

  // convert color space
  if (ImageUtil::IsColor(color_space) && color_space != "BGR") {
    ImageUtil::ConvertColor("BGR", image, color_space, image);
  }

  CHECK(image.isContinuous());
  const int c = ImageUtil::IsColor(color_space) ? 3 : 1;
  CHECK_EQ(c, image.channels());
  Shape image_shape({H, W, c});
  buffer->Resize(image_shape, DataType::kUInt8);
  CHECK_EQ(image_shape.elem_cnt(), buffer->nbytes());
  CHECK_EQ(image_shape.elem_cnt(), image.total() * image.elemSize());
  memcpy(buffer->mut_data<uint8_t>(), image.ptr(), image_shape.elem_cnt());
}

class RandCropGens final : public user_op::OpKernelState {
 public:
  explicit RandCropGens(int32_t size) : gens_(size) {}
  ~RandCropGens() = default;

  RandomCropGenerator* Get(int32_t idx) { return gens_.at(idx).get(); }

  void New(int32_t idx, AspectRatioRange aspect_ratio_range, AreaRange area_range, int64_t seed,
           int32_t num_attempts) {
    CHECK_LT(idx, gens_.size());
    gens_.at(idx).reset(
        new RandomCropGenerator(aspect_ratio_range, area_range, seed, num_attempts));
  }

 private:
  std::vector<std::shared_ptr<RandomCropGenerator>> gens_;
};

}  // namespace

class OFRecordImageDecoderRandomCropKernel final : public user_op::OpKernel {
 public:
  OFRecordImageDecoderRandomCropKernel() = default;
  ~OFRecordImageDecoderRandomCropKernel() override = default;

 private:
  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    int32_t num_attempts = ctx->Attr<int32_t>("num_attempts");
    CHECK(num_attempts >= 1);
    const std::vector<float>& random_aspect_ratio =
        ctx->Attr<std::vector<float>>("random_aspect_ratio");
    CHECK(random_aspect_ratio.size() == 2 && 0 < random_aspect_ratio.at(0)
          && random_aspect_ratio.at(0) <= random_aspect_ratio.at(1));
    const std::vector<float>& random_area = ctx->Attr<std::vector<float>>("random_area");
    CHECK(random_area.size() == 2 && 0 < random_area.at(0)
          && random_area.at(0) <= random_area.at(1));
    const user_op::TensorDesc* out_tensor_desc = ctx->TensorDesc4ArgNameAndIndex("out", 0);
    CHECK(out_tensor_desc->shape().NumAxes() == 1);
    int64_t batch_size = out_tensor_desc->shape().At(0);
    CHECK(batch_size > 0);
    int64_t seed = GetOpKernelRandomSeed(ctx);
    std::seed_seq seq{seed};
    std::vector<int> seeds(batch_size);
    seq.generate(seeds.begin(), seeds.end());

    std::shared_ptr<RandCropGens> crop_window_generators(new RandCropGens(batch_size));
    for (int32_t i = 0; i < batch_size; ++i) {
      crop_window_generators->New(i, {random_aspect_ratio.at(0), random_aspect_ratio.at(1)},
                                  {random_area.at(0), random_area.at(1)}, seeds.at(i),
                                  num_attempts);
    }
    return crop_window_generators;
  }

  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    auto* crop_window_generators = dynamic_cast<RandCropGens*>(state);
    user_op::Tensor* out_blob = ctx->Tensor4ArgNameAndIndex("out", 0);
    int64_t record_num = out_blob->shape().At(0);
    CHECK(record_num > 0);
    user_op::Tensor* in_blob = ctx->Tensor4ArgNameAndIndex("in", 0);
    CHECK_EQ(out_blob->shape(), in_blob->shape());
    OFRecord* records = in_blob->mut_dptr<OFRecord>();
    TensorBuffer* buffers = out_blob->mut_dptr<TensorBuffer>();
    const std::string& name = ctx->Attr<std::string>("name");
    const std::string& color_space = ctx->Attr<std::string>("color_space");

    MultiThreadLoop(record_num, [&](size_t i) {
      const OFRecord& record = *(records + i);
      TensorBuffer* buffer = buffers + i;
      RandomCropGenerator* gen = crop_window_generators->Get(i);
      DecodeRandomCropImageFromOneRecord(record, buffer, name, color_space, gen);
    });
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("ofrecord_image_decoder_random_crop")
    .SetCreateFn<OFRecordImageDecoderRandomCropKernel>()
    .SetIsMatchedHob(user_op::HobDeviceType() == DeviceType::kCPU
                     & user_op::HobDataType("in", 0) == DataType::kOFRecord
                     & user_op::HobDataType("out", 0) == DataType::kTensorBuffer);

class OFRecordImageDecoderKernel final : public user_op::OpKernel {
 public:
  OFRecordImageDecoderKernel() = default;
  ~OFRecordImageDecoderKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    user_op::Tensor* out_blob = ctx->Tensor4ArgNameAndIndex("out", 0);
    int64_t record_num = out_blob->shape().At(0);
    CHECK(record_num > 0);
    user_op::Tensor* in_blob = ctx->Tensor4ArgNameAndIndex("in", 0);
    CHECK_EQ(out_blob->shape(), in_blob->shape());
    OFRecord* records = in_blob->mut_dptr<OFRecord>();
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
    .SetIsMatchedHob(user_op::HobDeviceType() == DeviceType::kCPU
                     & user_op::HobDataType("in", 0) == DataType::kOFRecord
                     & user_op::HobDataType("out", 0) == DataType::kTensorBuffer);

}  // namespace oneflow
