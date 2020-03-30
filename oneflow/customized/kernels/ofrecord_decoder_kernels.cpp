#include "oneflow/core/framework/framework.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/common/blocking_counter.h"
#include "oneflow/core/common/tensor_buffer.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/thread/thread_manager.h"
#include "oneflow/customized/image/random_crop_generator.h"
#include "oneflow/customized/image/image_util.h"

#include <opencv2/opencv.hpp>

namespace oneflow {

template<typename T>
class OFRecordRawDecoderKernel final : public user_op::OpKernel {
 public:
  OFRecordRawDecoderKernel(user_op::KernelInitContext* ctx) : user_op::OpKernel(ctx) {
    auto_zero_padding_ = ctx->GetAttr<bool>("auto_zero_padding");
    dim1_varying_length_ = ctx->GetAttr<bool>("dim1_varying_length");
  }
  OFRecordRawDecoderKernel() = default;
  ~OFRecordRawDecoderKernel() = default;

 private:
  void DecodeOneRecord(const Feature& feature, T* dptr, int64_t sample_elem_cnt) {
    if (feature.has_bytes_list()) {
      CHECK_EQ(feature.bytes_list().value_size(), 1);
      const auto& value0 = feature.bytes_list().value(0);
      auto in_dptr = reinterpret_cast<const int8_t*>(value0.c_str());
      sample_elem_cnt = std::min<int64_t>(sample_elem_cnt, value0.size());
      CopyElem<int8_t, T>(in_dptr, dptr, sample_elem_cnt);
    }
#define DEFINE_ONE_ELIF(PbT, CppT)                                                                 \
  else if (feature.has_##PbT##_list()) {                                                           \
    const auto& list = feature.PbT##_list();                                                       \
    const CppT* in_dptr = list.value().data();                                                     \
    const int64_t padding_elem_num = auto_zero_padding_ ? sample_elem_cnt - list.value_size() : 0; \
    if (dim1_varying_length_ || auto_zero_padding_) {                                              \
      CHECK_LE(list.value_size(), sample_elem_cnt);                                                \
      sample_elem_cnt = list.value_size();                                                         \
    } else {                                                                                       \
      CHECK_EQ(sample_elem_cnt, list.value_size());                                                \
    }                                                                                              \
    CopyElem<CppT, T>(in_dptr, dptr, sample_elem_cnt);                                             \
    if (padding_elem_num > 0) {                                                                    \
      std::memset(dptr + sample_elem_cnt, 0, padding_elem_num * sizeof(T));                        \
    }                                                                                              \
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

  void DecodePartRecords(int32_t part_id, int32_t part_num, OFRecord* records, int64_t record_num,
                         T* out_dptr, int64_t sample_elem_cnt, const std::string& name) {
    BalancedSplitter bs(record_num, part_num);
    Range range = bs.At(part_id);
    FOR_RANGE(int32_t, i, range.begin(), range.end()) {
      const OFRecord& record = *(records + i);
      T* dptr = out_dptr + i * sample_elem_cnt;
      CHECK(record.feature().find(name) != record.feature().end())
          << "Field " << name << " not found";
      const Feature& feature = record.feature().at(name);
      DecodeOneRecord(feature, dptr, sample_elem_cnt);
    }
  }

  void Compute(user_op::KernelContext* ctx) override {
    user_op::Tensor* in_blob = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out_blob = ctx->Tensor4ArgNameAndIndex("out", 0);
    // TODO(chengcheng): remove record num in record blob, fix by shape elem cnt
    int64_t record_num = in_blob->shape().At(0);
    int64_t sample_elem_cnt = out_blob->shape().Count(1);
    CHECK(record_num > 0);
    OFRecord* records = in_blob->mut_dptr<OFRecord>();
    T* out_dptr = out_blob->mut_dptr<T>();
    std::string name = ctx->GetAttr<std::string>("name");

    ThreadPool* thread_pool = Global<ThreadMgr>::Get()->compute_thread_pool();
    int32_t thread_num = thread_pool->thread_num();
    int32_t part_num = std::min(static_cast<int32_t>(record_num), thread_num);
    BlockingCounter bc(part_num);
    FOR_RANGE(int32_t, part_id, 0, part_num) {
      thread_pool->AddWork([&bc, part_id, part_num, records, record_num, out_dptr, sample_elem_cnt,
                            &name, this]() {
        DecodePartRecords(part_id, part_num, records, record_num, out_dptr, sample_elem_cnt, name);
        bc.Decrease();
      });
    }
    bc.WaitUntilCntEqualZero();
  }

  bool auto_zero_padding_;
  bool dim1_varying_length_;
};

#define REGISTER_RAW_DECODER_KERNEL(dtype)                                                         \
  REGISTER_USER_KERNEL("OFRecordRawDecoder")                                                       \
      .SetCreateFn([](user_op::KernelInitContext* ctx) {                                           \
        return new OFRecordRawDecoderKernel<dtype>(ctx);                                           \
      })                                                                                           \
      .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {                                 \
        const user_op::TensorDesc* in_tensor = ctx.TensorDesc4ArgNameAndIndex("in", 0);            \
        const user_op::TensorDesc* out_tensor = ctx.TensorDesc4ArgNameAndIndex("out", 0);          \
        if (ctx.device_type() == DeviceType::kCPU && in_tensor->data_type() == DataType::kOFRecord \
            && out_tensor->data_type() == GetDataType<dtype>::value) {                             \
          return true;                                                                             \
        }                                                                                          \
        return false;                                                                              \
      });

REGISTER_RAW_DECODER_KERNEL(char)
REGISTER_RAW_DECODER_KERNEL(float)
REGISTER_RAW_DECODER_KERNEL(double)
REGISTER_RAW_DECODER_KERNEL(int8_t)
REGISTER_RAW_DECODER_KERNEL(int32_t)
REGISTER_RAW_DECODER_KERNEL(int64_t)
REGISTER_RAW_DECODER_KERNEL(uint8_t)

class OFRecordImageDecoderRandomCropKernel final : public user_op::OpKernel {
 public:
  OFRecordImageDecoderRandomCropKernel(user_op::KernelInitContext* ctx) : user_op::OpKernel(ctx) {
    int32_t num_attempts = ctx->GetAttr<int32_t>("num_attempts");
    CHECK(num_attempts >= 1);
    std::vector<float> random_aspect_ratio =
        ctx->GetAttr<std::vector<float>>("random_aspect_ratio");
    CHECK(random_aspect_ratio.size() == 2 && 0 < random_aspect_ratio.at(0)
          && random_aspect_ratio.at(0) <= random_aspect_ratio.at(1));
    std::vector<float> random_area = ctx->GetAttr<std::vector<float>>("random_area");
    CHECK(random_area.size() == 2 && 0 < random_area.at(0)
          && random_area.at(0) <= random_area.at(1));
    const user_op::TensorDesc* out_tensor_desc = ctx->TensorDesc4ArgNameAndIndex("out", 0);
    CHECK(out_tensor_desc->shape().NumAxes() == 1);
    int64_t batch_size = out_tensor_desc->shape().At(0);
    CHECK(batch_size > 0);
    int64_t seed = ctx->GetAttr<int64_t>("seed");
    if (seed == -1) { seed = NewRandomSeed(); }
    CHECK(seed >= 0);
    std::seed_seq seq{seed};
    std::vector<int> seeds(batch_size);
    seq.generate(seeds.begin(), seeds.end());

    crop_window_generators_.resize(batch_size);
    for (int32_t i = 0; i < batch_size; ++i) {
      crop_window_generators_.at(i).reset(new RandomCropGenerator(
          {random_aspect_ratio.at(0), random_aspect_ratio.at(1)},
          {random_area.at(0), random_area.at(1)}, seeds.at(i), num_attempts));
    }
  }
  OFRecordImageDecoderRandomCropKernel() = default;
  ~OFRecordImageDecoderRandomCropKernel() = default;

 private:
  void DecodePartRecords(int32_t part_id, int32_t part_num, OFRecord* records, int64_t record_num,
                         TensorBuffer* buffers, const std::string& name,
                         const std::string& color_space) {
    BalancedSplitter bs(record_num, part_num);
    Range range = bs.At(part_id);
    FOR_RANGE(int32_t, i, range.begin(), range.end()) {
      const OFRecord& record = *(records + i);
      TensorBuffer* buffer = buffers + i;
      CHECK(record.feature().find(name) != record.feature().end())
          << "Field " << name << " not found";
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
      CHECK(image.data != nullptr);
      cv::Mat image_roi;
      CropWindow crop = crop_window_generators_.at(i)->GenerateCropWindow({H, W});
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
  }

  void Compute(user_op::KernelContext* ctx) override {
    user_op::Tensor* out_blob = ctx->Tensor4ArgNameAndIndex("out", 0);
    // TODO(chengcheng): remove record num in record blob, fix by shape elem cnt
    int64_t record_num = out_blob->shape().At(0);
    CHECK(record_num > 0);
    user_op::Tensor* in_blob = ctx->Tensor4ArgNameAndIndex("in", 0);
    CHECK_EQ(out_blob->shape(), in_blob->shape());
    OFRecord* records = in_blob->mut_dptr<OFRecord>();
    TensorBuffer* buffers = out_blob->mut_dptr<TensorBuffer>();
    std::string name = ctx->GetAttr<std::string>("name");
    std::string color_space = ctx->GetAttr<std::string>("color_space");

    ThreadPool* thread_pool = Global<ThreadMgr>::Get()->compute_thread_pool();
    int32_t thread_num = thread_pool->thread_num();
    int32_t part_num = std::min(static_cast<int32_t>(record_num), thread_num);
    BlockingCounter bc(part_num);
    FOR_RANGE(int32_t, part_id, 0, part_num) {
      thread_pool->AddWork(
          [&bc, part_id, part_num, records, record_num, buffers, &name, &color_space, this]() {
            DecodePartRecords(part_id, part_num, records, record_num, buffers, name, color_space);
            bc.Decrease();
          });
    }
    bc.WaitUntilCntEqualZero();
  }

  std::vector<std::shared_ptr<RandomCropGenerator>> crop_window_generators_;
};

REGISTER_USER_KERNEL("OFRecordImageDecoderRandomCrop")
    .SetCreateFn([](user_op::KernelInitContext* ctx) {
      return new OFRecordImageDecoderRandomCropKernel(ctx);
    })
    .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {
      const user_op::TensorDesc* in_tensor = ctx.TensorDesc4ArgNameAndIndex("in", 0);
      const user_op::TensorDesc* out_tensor = ctx.TensorDesc4ArgNameAndIndex("out", 0);
      if (ctx.device_type() == DeviceType::kCPU && in_tensor->data_type() == DataType::kOFRecord
          && out_tensor->data_type() == DataType::kTensorBuffer) {
        return true;
      }
      return false;
    });

}  // namespace oneflow
