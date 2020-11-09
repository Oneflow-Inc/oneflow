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
#include "oneflow/user/summary/event_writer_helper.h"
#include "oneflow/user/summary/env_time.h"
#include "oneflow/user/summary/events_writer.h"
#include "oneflow/user/summary/histogram.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/summary/summary.pb.h"
#include "oneflow/core/summary/event.pb.h"

#include <png.h>
#include <zlib.h>
#include <memory>
#include <type_traits>
#define USER_LIBPNG_VER_STRING "1.6.24"

namespace oneflow {

namespace summary {

const char* kScalarPluginName = "scalars";
const char* kHistogramPluginName = "histograms";
const char* kImagePluginName = "images";

void SetPluginData(SummaryMetadata* metadata, const char* name) {
  if (metadata->plugin_data().plugin_name().empty()) {
    metadata->mutable_plugin_data()->set_plugin_name(name);
  }
}

Maybe<void> FillScalarInSummary(const float& value, const std::string& tag, Summary* s) {
  SummaryMetadata metadata;
  SetPluginData(&metadata, kScalarPluginName);
  Summary::Value* v = s->add_value();
  v->set_tag(tag);
  *v->mutable_metadata() = metadata;
  v->set_simple_value(value);
  return Maybe<void>::Ok();
}

template<typename T>
Maybe<void> FillHistogramInSummary(const user_op::Tensor& value, const std::string& tag,
                                   Summary* s) {
  SummaryMetadata metadata;
  SetPluginData(&metadata, kHistogramPluginName);
  Summary::Value* v = s->add_value();
  v->set_tag(tag);
  *v->mutable_metadata() = metadata;
  summary::Histogram histo;
  for (int64_t i = 0; i < value.shape().elem_cnt(); i++) {
    double double_val = value.dptr<T>()[i];
    histo.AppendValue(double_val);
  }
  histo.AppendToProto(v->mutable_histo());
  return Maybe<void>::Ok();
}

void WriteImageDataFn(png_structp png_ptr, png_bytep data, png_size_t length) {
  std::string* const s = reinterpret_cast<std::string*>(png_get_io_ptr(png_ptr));
  s->append(reinterpret_cast<const char*>(data), length);
}

bool WriteImageToBuffer(const uint8_t* image, int width, int height, int depth,
                        std::string* png_string) {
  CHECK_NOTNULL(image);
  CHECK_NOTNULL(png_string);
  if (width == 0 || height == 0) return false;
  png_string->resize(0);
  png_infop info_ptr = nullptr;
  png_structp png_ptr = png_create_write_struct(USER_LIBPNG_VER_STRING, 0, 0, 0);
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
  switch (depth) {
    case 1: color_type = PNG_COLOR_TYPE_GRAY; break;
    case 2: color_type = PNG_COLOR_TYPE_GRAY_ALPHA; break;
    case 3: color_type = PNG_COLOR_TYPE_RGB; break;
    case 4: color_type = PNG_COLOR_TYPE_RGB_ALPHA; break;
    default: png_destroy_write_struct(&png_ptr, &info_ptr); return false;
  }
  const int bit_depth = 8;
  png_set_write_fn(png_ptr, png_string, WriteImageDataFn, nullptr);
  png_set_compression_level(png_ptr, Z_DEFAULT_COMPRESSION);
  png_set_compression_mem_level(png_ptr, MAX_MEM_LEVEL);
  png_set_IHDR(png_ptr, info_ptr, width, height, bit_depth, color_type, PNG_INTERLACE_NONE,
               PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
  png_write_info(png_ptr, info_ptr);
  png_byte* row = reinterpret_cast<png_byte*>(const_cast<uint8_t*>(image));
  int row_bytes = width * depth;
  for (; height--; row += row_bytes) png_write_row(png_ptr, row);
  png_write_end(png_ptr, nullptr);
  png_destroy_write_struct(&png_ptr, &info_ptr);
  return true;
}

Maybe<void> FillImageInSummary(const user_op::Tensor& tensor, const std::string& tag, Summary* s) {
  SummaryMetadata metadata;
  SetPluginData(&metadata, kImagePluginName);
  if (!(tensor.shape().NumAxes() == 4
        && (tensor.shape().At(3) == 1 || tensor.shape().At(3) == 3 || tensor.shape().At(3) == 4))) {
    UNIMPLEMENTED();
  }
  if (!(tensor.shape().At(0) < (1LL << 31) && tensor.shape().At(1) < (1LL << 31)
        && tensor.shape().At(2) < (1LL << 31)
        && (tensor.shape().At(1) * tensor.shape().At(2)) < (1LL << 29))) {
    UNIMPLEMENTED();
  }
  const int64_t batch_size = static_cast<int64_t>(tensor.shape().At(0));
  const int64_t h = static_cast<int64_t>(tensor.shape().At(1));
  const int64_t w = static_cast<int64_t>(tensor.shape().At(2));
  const int64_t hw = h * w;
  const int64_t depth = static_cast<int64_t>(tensor.shape().At(3));
  if (tensor.data_type() == DataType::kUInt8) {
    auto ith_image = [&tensor, hw, depth](int i) {
      auto images = tensor.dptr<uint8_t>();
      uint8_t* image_i = (uint8_t*)malloc(sizeof(uint8_t) * hw * depth);
      memcpy(image_i, images + i * hw * depth, hw * depth);
      return image_i;
    };
    for (int i = 0; i < batch_size; ++i) {
      Summary::Value* v = s->add_value();
      *v->mutable_metadata() = metadata;
      if (batch_size == 1) {
        v->set_tag(tag);
      } else {
        v->set_tag(tag + std::to_string(i));
      }
      Image* si = v->mutable_image();
      si->set_height(h);
      si->set_width(w);
      si->set_colorspace(depth);
      auto image = ith_image(i);
      if (!WriteImageToBuffer(image, w, h, depth, si->mutable_encoded_image_string()))
        UNIMPLEMENTED();
    }
  }
  return Maybe<void>::Ok();
}

template<typename T>
struct EventWriterHelper<DeviceType::kCPU, T> {
  static void WritePbToFile(int64_t step, const std::string& value) {
    std::unique_ptr<Event> e{new Event};
    Summary sum;
    TxtString2PbMessage(value, &sum);
    e->set_step(step);
    e->set_wall_time(GetWallTime());
    *e->mutable_summary() = sum;
    Global<EventsWriter>::Get()->AppendQueue(std::move(e));
  }

  static void WriteScalarToFile(int64_t step, float value, const std::string& tag) {
    std::unique_ptr<Event> e{new Event};
    e->set_step(step);
    e->set_wall_time(GetWallTime());
    FillScalarInSummary(value, tag, e->mutable_summary());
    Global<EventsWriter>::Get()->AppendQueue(std::move(e));
  }

  static void WriteHistogramToFile(int64_t step, const user_op::Tensor& value,
                                   const std::string& tag) {
    std::unique_ptr<Event> e{new Event};
    e->set_step(step);
    e->set_wall_time(GetWallTime());
    FillHistogramInSummary<T>(value, tag, e->mutable_summary());
    Global<EventsWriter>::Get()->AppendQueue(std::move(e));
  }

  static void WriteImageToFile(int64_t step, const user_op::Tensor& tensor,
                               const std::string& tag) {
    std::unique_ptr<Event> e{new Event};
    e->set_step(step);
    e->set_wall_time(GetWallTime());
    FillImageInSummary(tensor, tag, e->mutable_summary());
    Global<EventsWriter>::Get()->AppendQueue(std::move(e));
  }
};

#define INSTANTIATE_EVENT_WRITE_HELPER_CPU(dtype) \
  template struct EventWriterHelper<DeviceType::kCPU, dtype>;

INSTANTIATE_EVENT_WRITE_HELPER_CPU(float)
INSTANTIATE_EVENT_WRITE_HELPER_CPU(double)
INSTANTIATE_EVENT_WRITE_HELPER_CPU(int32_t)
INSTANTIATE_EVENT_WRITE_HELPER_CPU(int64_t)
INSTANTIATE_EVENT_WRITE_HELPER_CPU(uint8_t)
INSTANTIATE_EVENT_WRITE_HELPER_CPU(int8_t)

}  // namespace summary

}  // namespace oneflow
