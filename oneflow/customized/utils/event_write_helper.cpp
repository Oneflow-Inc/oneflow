#include "oneflow/customized/utils/event_write_helper.h"
#include "oneflow/core/common/device_type.pb.h"
#include "oneflow/customized/utils/event.pb.h"
#include "oneflow/customized/utils/env_time.h"
#include "oneflow/customized/utils/events_writer.h"
#include "oneflow/customized/utils/histogram.h"
#include "oneflow/customized/utils/summary.pb.h"
#include "oneflow/core/common/protobuf.h"

#include <png.h>
#include <zlib.h>
#include <memory>
#include <opencv2/opencv.hpp>
#include <type_traits>
#define PNG_LIBPNG_VER_STRING "1.6.24"

namespace oneflow {

namespace {

const char* kScalarPluginName = "scalars";
const char* kHistogramPluginName = "histograms";
const char* kImagePluginName = "images";
const char* kAudioPluginName = "audio";

void SetPluginData(SummaryMetadata* metadata, const char* name) {
  if (metadata->plugin_data().plugin_name().empty()) {
    metadata->mutable_plugin_data()->set_plugin_name(name);
  }
}

Maybe<void> AddScalarToSummary(const float& value, const std::string& tag, Summary* s) {
  SummaryMetadata metadata;
  SetPluginData(&metadata, kScalarPluginName);
  Summary::Value* v = s->add_value();
  v->set_tag(tag);
  *v->mutable_metadata() = metadata;
  v->set_simple_value(value);
  return Maybe<void>::Ok();
}

template<typename T>
Maybe<void> AddHistogramToSummary(const user_op::Tensor& value, const std::string& tag,
                                  Summary* s) {
  SummaryMetadata metadata;
  SetPluginData(&metadata, kHistogramPluginName);
  Summary::Value* v = s->add_value();
  v->set_tag(tag);
  *v->mutable_metadata() = metadata;
  histogram::Histogram histo;
  for (int64_t i = 0; i < value.shape().elem_cnt(); i++) {
    double double_val = value.dptr<T>()[i];
    histo.AppendValue(double_val);
  }
  histo.AppendToProto(v->mutable_histo());
  return Maybe<void>::Ok();
}

void StringWriter(png_structp png_ptr, png_bytep data, png_size_t length) {
  std::string* const s = reinterpret_cast<std::string*>(png_get_io_ptr(png_ptr));
  s->append(reinterpret_cast<const char*>(data), length);
}

void StringWriterFlush(png_structp png_ptr) {}

bool WriteImageToBuffer(const uint8_t* image, int width, int height, int row_bytes,
                        int num_channels, int channel_bits, int compression,
                        std::string* png_string) {
  CHECK_NOTNULL(image);
  CHECK_NOTNULL(png_string);
  if (width == 0 || height == 0) return false;

  png_string->resize(0);
  png_infop info_ptr = nullptr;
  png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, 0, 0, 0);
  if (png_ptr == nullptr) return false;
  if (setjmp(png_jmpbuf(png_ptr))) {
    png_destroy_write_struct(&png_ptr, info_ptr ? &info_ptr : nullptr);
    return false;
  }
  info_ptr = png_create_info_struct(png_ptr);
  if (info_ptr == nullptr) {
    png_destroy_write_struct(&png_ptr, nullptr);
    return false;
  }
  int color_type = -1;
  switch (num_channels) {
    case 1: color_type = PNG_COLOR_TYPE_GRAY; break;
    case 2: color_type = PNG_COLOR_TYPE_GRAY_ALPHA; break;
    case 3: color_type = PNG_COLOR_TYPE_RGB; break;
    case 4: color_type = PNG_COLOR_TYPE_RGB_ALPHA; break;
    default: png_destroy_write_struct(&png_ptr, &info_ptr); return false;
  }
  png_set_write_fn(png_ptr, png_string, StringWriter, StringWriterFlush);
  if (compression < 0) compression = Z_DEFAULT_COMPRESSION;
  png_set_compression_level(png_ptr, compression);
  png_set_compression_mem_level(png_ptr, MAX_MEM_LEVEL);
  png_set_IHDR(png_ptr, info_ptr, width, height, channel_bits, color_type, PNG_INTERLACE_NONE,
               PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
  png_write_info(png_ptr, info_ptr);
  png_byte* row = reinterpret_cast<png_byte*>(const_cast<uint8_t*>(image));
  for (; height--; row += row_bytes) {
    png_write_row(png_ptr, row);
    auto j = row;
  }

  png_write_end(png_ptr, nullptr);
  png_destroy_write_struct(&png_ptr, &info_ptr);
  return true;
}

Maybe<void> AddImageToSummary(const user_op::Tensor* tensor, const std::string& tag, Summary* s) {
  SummaryMetadata metadata;
  SetPluginData(&metadata, kImagePluginName);
  if (!(tensor->shape().NumAxes() == 4
        && (tensor->shape().At(3) == 1 || tensor->shape().At(3) == 3
            || tensor->shape().At(3) == 4))) {
    UNIMPLEMENTED();
  }
  if (!(tensor->shape().At(0) < (1LL << 31) && tensor->shape().At(1) < (1LL << 31)
        && tensor->shape().At(2) < (1LL << 31)
        && (tensor->shape().At(1) * tensor->shape().At(2)) < (1LL << 29))) {
    UNIMPLEMENTED();
  }
  const int64_t batch_size = static_cast<int64_t>(tensor->shape().At(0));
  const int64_t h = static_cast<int64_t>(tensor->shape().At(1));
  const int64_t w = static_cast<int64_t>(tensor->shape().At(2));
  const int64_t hw = h * w;
  const int64_t depth = static_cast<int64_t>(tensor->shape().At(3));
  const int channel_bits = 8;
  const int compression = -1;
  if (tensor->data_type() == DataType::kUInt8 || tensor->data_type() == DataType::kInt8
      || tensor->data_type() == DataType::kChar) {
    auto ith_image = [tensor, hw, depth](int i) {
      auto values = tensor->dptr<uint8_t>();
      uint8_t* image_ptr = (uint8_t*)malloc(sizeof(uint8_t) * hw * depth);
      FOR_RANGE(int, j, 0, hw* depth) { image_ptr[j] = values[i * hw * depth + j]; }
      return image_ptr;
    };
    for (int i = 0; i < batch_size; ++i) {
      Summary::Value* v = s->add_value();
      *v->mutable_metadata() = metadata;
      v->set_tag(tag + std::to_string(i));
      Image* si = v->mutable_image();
      si->set_height(h);
      si->set_width(w);
      si->set_colorspace(depth);
      auto image = ith_image(i);
      if (!WriteImageToBuffer(image, w, h, w * depth, depth, channel_bits, compression,
                              si->mutable_encoded_image_string()))
        UNIMPLEMENTED();
    }
  }
  return Maybe<void>::Ok();
}

}  // namespace

template<typename T>
struct EventWriteHelper<DeviceType::kCPU, T> {
  static void WritePbToFile(int64_t step, const std::string& value) {
    Event e;
    Summary sum;
    TxtString2PbMessage(value, &sum);
    e.set_step(step);
    e.set_wall_time(envtime::GetWallTime());
    *e.mutable_summary() = sum;
    Global<EventsWriter>::Get()->WriteEvent(e);
  }

  static void WriteScalarToFile(int64_t step, float value, const std::string& tag) {
    Event e;
    e.set_step(step);
    e.set_wall_time(envtime::GetWallTime());
    AddScalarToSummary(value, tag, e.mutable_summary());
    Global<EventsWriter>::Get()->WriteEvent(e);
  }

  static void WriteHistogramToFile(int64_t step, const user_op::Tensor& value,
                                   const std::string& tag) {
    Event e;
    e.set_step(step);
    e.set_wall_time(envtime::GetWallTime());
    AddHistogramToSummary<T>(value, tag, e.mutable_summary());
    Global<EventsWriter>::Get()->WriteEvent(e);
  }

  static void WriteImageToFile(int64_t step, const user_op::Tensor* tensor,
                               const std::string& tag) {
    Event e;
    e.set_step(step);
    e.set_wall_time(envtime::GetWallTime());
    AddImageToSummary(tensor, tag, e.mutable_summary());
    return Global<EventsWriter>::Get()->WriteEvent(e);
  }
};

#define INSTANTIATE_EVENT_WRITE_HELPER_CPU(dtype) \
  template struct EventWriteHelper<DeviceType::kCPU, dtype>;

INSTANTIATE_EVENT_WRITE_HELPER_CPU(float)
INSTANTIATE_EVENT_WRITE_HELPER_CPU(double)
INSTANTIATE_EVENT_WRITE_HELPER_CPU(int32_t)
INSTANTIATE_EVENT_WRITE_HELPER_CPU(int64_t)
INSTANTIATE_EVENT_WRITE_HELPER_CPU(uint8_t)
INSTANTIATE_EVENT_WRITE_HELPER_CPU(int8_t)

}  // namespace oneflow