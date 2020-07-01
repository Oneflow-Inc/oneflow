#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/common/shape.h"
#include "oneflow/core/common/shape_vec.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/customized/utils/event.pb.h"
#include "oneflow/customized/utils/events_writer.h"
#include "oneflow/customized/utils/summary.pb.h"
#include "oneflow/customized/utils/env_time.h"
#include "oneflow/customized/utils/histogram.h"
#include "oneflow/customized/utils/tensor.pb.h"
#include "oneflow/core/common/protobuf.h"

#include <sys/time.h>
#include <time.h>
#include <memory>

#include <png.h>
#include <zlib.h>
#include <opencv2/opencv.hpp>
#define PNG_LIBPNG_VER_STRING "1.6.24"

namespace oneflow {

namespace {

const char* kScalarPluginName = "scalars";
const char* kImagePluginName = "images";
const char* kAudioPluginName = "audio";
const char* kHistogramPluginName = "histograms";
const char* kTextPluginName = "text";

void PatchPluginName(SummaryMetadata* metadata, const char* name) {
  if (metadata->plugin_data().plugin_name().empty()) {
    metadata->mutable_plugin_data()->set_plugin_name(name);
  }
}

void StringWriter(png_structp png_ptr, png_bytep data, png_size_t length) {
  std::string* const s = reinterpret_cast<std::string*>(png_get_io_ptr(png_ptr));
  s->append(reinterpret_cast<const char*>(data), length);
}

void StringWriterFlush(png_structp png_ptr) {}

void ErrorHandler(png_structp png_ptr, png_const_charp msg) {
  VLOG(1) << "PNG error: " << msg;
  longjmp(png_jmpbuf(png_ptr), 1);
}

void WarningHandler(png_structp png_ptr, png_const_charp msg) {
  LOG(WARNING) << "PNG warning: " << msg;
}
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
  // There used to be a call to png_set_filter here turning off filtering
  // entirely, but it produced pessimal compression ratios.  I'm not sure
  // why it was there.
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

Maybe<void> AddImageToSummary(
    const user_op::Tensor* tensor, const std::string& tag,
    //                              const int64_t max_images, const user_op::Tensor* bad_color,
    Summary* s) {
  SummaryMetadata metadata;
  PatchPluginName(&metadata, kImagePluginName);
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
    //  const int N = std::min(max_images, batch_size);
    const int N = batch_size;
    auto ith_image = [tensor, batch_size, hw, depth](int i) {
      auto values = tensor->dptr<uint8_t>();
      uint8_t* image_ptr = (uint8_t*)malloc(sizeof(uint8_t) * hw * depth);
      FOR_RANGE(int, j, 0, hw* depth) { image_ptr[j] = values[i * hw * depth + j]; }
      return image_ptr;
    };
    for (int i = 0; i < N; ++i) {
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
      // WriteImageToBuffer(image, w, h, w * depth, depth, channel_bits, compression,
      //                   si->mutable_encoded_image_string());
    }
  }
  return Maybe<void>::Ok();
}

Maybe<void> AddScalarToSummary(const float& value, const std::string& tag, Summary* s) {
  SummaryMetadata metadata;
  PatchPluginName(&metadata, kScalarPluginName);
  Summary::Value* v = s->add_value();
  v->set_tag(tag);
  *v->mutable_metadata() = metadata;
  v->set_simple_value(value);
  return Maybe<void>::Ok();
}

void WriteImage(const int64_t step, const user_op::Tensor* tensor, const std::string& tag) {
  //, const int64_t max_images, const user_op::Tensor* bad_color) {
  Event e;
  e.set_step(step);
  e.set_wall_time(envtime::GetWallTime());
  //  AddImageToSummary(tensor, tag, max_images, bad_color, e.mutable_summary());
  AddImageToSummary(tensor, tag, e.mutable_summary());
  return Global<EventsWriter>::Get()->WriteEvent(e);
}

void WriteScalar(int64_t step, const float& value, const std::string& tag) {
  Event e;
  e.set_step(step);
  e.set_wall_time(envtime::GetWallTime());
  AddScalarToSummary(value, tag, e.mutable_summary());
  // Global<EventsWriter>::Get()->Initialize("/home/zjhushengjian/oneflow", "laoxu");
  Global<EventsWriter>::Get()->WriteEvent(e);
}

template<typename T>
Maybe<void> AddHistogramToSummary(const user_op::Tensor& value, const std::string& tag,
                                  Summary* s) {
  SummaryMetadata metadata;
  PatchPluginName(&metadata, kHistogramPluginName);
  Summary::Value* v = s->add_value();
  v->set_tag(tag);
  *v->mutable_metadata() = metadata;
  histogram::Histogram histo;
  for (int64_t i = 0; i < value.shape().elem_cnt(); i++) {
    double double_val = value.dptr<T>()[i];
    // float double_val = value.dptr<float>()[i];
    // TF_RETURN_IF_ERROR(TensorValueAt<double>(t, i, &double_val));

    // if (Eigen::numext::isnan(double_val)) {
    //   return errors::InvalidArgument("Nan in summary histogram for: ", tag);
    // } else if (Eigen::numext::isinf(double_val)) {
    //   return errors::InvalidArgument("Infinity in summary histogram for: ", tag);
    // }
    histo.AppendValue(double_val);
  }
  histo.AppendToProto(v->mutable_histo());
  return Maybe<void>::Ok();
}

template<typename T>
struct Helper {
  typedef google::protobuf::RepeatedField<T> RepeatedFieldType;
};

template<typename T>
struct ProtoHelper {};

// #define PROTO_TRAITS(T, F, N)                                                              \
//   template<>                                                                               \
//   struct ProtoHelper<T> {                                                                  \
//     typedef Helper<F>::RepeatedFieldType FieldType;                                        \
//     static FieldType::const_iterator Begin(const TensorProto& proto) {                     \
//       return proto.N##_val().begin();                                                      \
//     }                                                                                      \
//     static size_t NumElements(const TensorProto& proto) { return proto.N##_val().size(); } \
//     static void Fill(const T* data, size_t n, TensorProto* proto) {                        \
//       typename ProtoHelper<T>::FieldType copy(data, data + n);                             \
//       proto->mutable_##N##_val()->Swap(&copy);                                             \
//     }                                                                                      \
//   };
// PROTO_TRAITS(float, float, float);
// PROTO_TRAITS(double, double, double);
// PROTO_TRAITS(int32_t, int32_t, int);
// //PROTO_TRAITS(uint8_t, int32_t, int);
// PROTO_TRAITS(uint16_t, int32_t, int);
// PROTO_TRAITS(int16_t, int32_t, int);
// PROTO_TRAITS(int8_t, int32_t, int);
// PROTO_TRAITS(bool, bool, bool);

// #define SINGLE_ARG(...) __VA_ARGS__
// #define CASE(TYPE, STMTS)             \
//   case DataTypeToEnum<TYPE>::value: { \
//     typedef TYPE T;                   \
//     STMTS;                            \
//     break;                            \
//   }
// #define CASES_WITH_DEFAULT(TYPE_ENUM, STMTS, INVALID, DEFAULT) \
//   switch (TYPE_ENUM) {                                         \
//     CASE(float, SINGLE_ARG(STMTS))                             \
//     CASE(double, SINGLE_ARG(STMTS))                            \
//     CASE(int32, SINGLE_ARG(STMTS))                             \
//     CASE(uint8, SINGLE_ARG(STMTS))                             \
//     CASE(uint16, SINGLE_ARG(STMTS))                            \
//     CASE(uint32, SINGLE_ARG(STMTS))                            \
//     CASE(uint64, SINGLE_ARG(STMTS))                            \
//     CASE(int16, SINGLE_ARG(STMTS))                             \
//     CASE(int8, SINGLE_ARG(STMTS))                              \
//     CASE(tstring, SINGLE_ARG(STMTS))                           \
//     CASE(complex64, SINGLE_ARG(STMTS))                         \
//     CASE(complex128, SINGLE_ARG(STMTS))                        \
//     CASE(int64, SINGLE_ARG(STMTS))                             \
//     CASE(bool, SINGLE_ARG(STMTS))                              \
//     CASE(qint32, SINGLE_ARG(STMTS))                            \
//     CASE(quint8, SINGLE_ARG(STMTS))                            \
//     CASE(qint8, SINGLE_ARG(STMTS))                             \
//     CASE(quint16, SINGLE_ARG(STMTS))                           \
//     CASE(qint16, SINGLE_ARG(STMTS))                            \
//     CASE(bfloat16, SINGLE_ARG(STMTS))                          \
//     CASE(Eigen::half, SINGLE_ARG(STMTS))                       \
//     CASE(ResourceHandle, SINGLE_ARG(STMTS))                    \
//     CASE(Variant, SINGLE_ARG(STMTS))                           \
//     case DT_INVALID:                                           \
//       INVALID;                                                 \
//       break;                                                   \
//     default:                                                   \
//       DEFAULT;                                                 \
//       break;                                                   \
//   }

// #define CASES(TYPE_ENUM, STMTS)                                      \
//   CASES_WITH_DEFAULT(TYPE_ENUM, STMTS, LOG(FATAL) << "Type not set"; \
//                      , LOG(FATAL) << "Unexpected type: " << TYPE_ENUM;)

template<>
struct ProtoHelper<std::string> {
  // static FieldType::const_iterator Begin(const TensorProto& proto) {
  //   return proto.string_val().begin();
  // }
  // static size_t NumElements(const TensorProto& proto) { return proto.string_val().size(); }
  static void Fill(const std::string* data, size_t n, TensorProto* proto) {
    typename google::protobuf::RepeatedPtrField<std::string> copy(data->begin(), data->end());
    proto->mutable_string_val()->Swap(&copy);
  }
};

#define CASES(TYPE_ENUM, STMTS)                                      \
  CASES_WITH_DEFAULT(TYPE_ENUM, STMTS, LOG(FATAL) << "Type not set"; \
                     , LOG(FATAL) << "Unexpected type: " << TYPE_ENUM;)

void ToProtoField(const user_op::Tensor& tensor, TensorProto* out) {
  int8_t* ptr = const_cast<int8_t*>(tensor.dptr<int8_t>());
  char* x = reinterpret_cast<char*>(ptr);
  const std::string d = x;
  ProtoHelper<std::string>::Fill(&d, tensor.shape().elem_cnt(), out);
}

void AsProto(TensorShapeProto* proto, const ShapeView& shape) {
  proto->Clear();
  DimVector dim_vec;
  shape.ToDimVector(&dim_vec);
  for (int i = 0; i < shape.NumAxes(); i++) { proto->add_dim()->set_size(dim_vec[i]); }
}

void AsProtoField(TensorProto* proto, const user_op::Tensor& tensor) {
  proto->Clear();
  AsProto(proto->mutable_tensor_shape(), tensor.shape());
  proto->set_dtype(TDataType::DT_STRING);
  ToProtoField(tensor, proto);
}

void AddTextToSummary(const user_op::Tensor& value, const std::string& tag, Summary* s) {
  SummaryMetadata metadata;
  PatchPluginName(&metadata, kTextPluginName);
  Summary::Value* v = s->add_value();
  v->set_tag(tag);
  *v->mutable_metadata() = metadata;
  if (value.data_type() == DataType::kInt8) { AsProtoField(v->mutable_tensor(), value); }
}

template<typename T>
void WriteHistogram(const float& step, const user_op::Tensor& value, const std::string& tag) {
  Event e;
  e.set_step(step);
  e.set_wall_time(envtime::GetWallTime());
  AddHistogramToSummary<T>(value, tag, e.mutable_summary());
  Global<EventsWriter>::Get()->WriteEvent(e);
}

void WriteText(const float& step, const user_op::Tensor& value, const std::string& tag) {
  Event e;
  e.set_step(step);
  e.set_wall_time(envtime::GetWallTime());
  AddTextToSummary(value, tag, e.mutable_summary());
  Global<EventsWriter>::Get()->WriteEvent(e);
}

template<typename T>
class WriteScalarOp final : public user_op::OpKernel {
 public:
  WriteScalarOp() = default;
  ~WriteScalarOp() = default;

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
    WriteScalar(istep[0], static_cast<double>(tvalue[0]), reinterpret_cast<char*>(ctag));
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return true; }
};

#define REGISTER_SCALAR_USER_KERNEL(dtype)                                            \
  REGISTER_USER_KERNEL("write_scalar")                                                \
      .SetCreateFn<WriteScalarOp<dtype>>()                                            \
      .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {                    \
        const user_op::TensorDesc* in_desc = ctx.TensorDesc4ArgNameAndIndex("in", 0); \
        return in_desc->data_type() == GetDataType<dtype>::value;                     \
      });

REGISTER_SCALAR_USER_KERNEL(double)
REGISTER_SCALAR_USER_KERNEL(float)
REGISTER_SCALAR_USER_KERNEL(int64_t)
REGISTER_SCALAR_USER_KERNEL(int32_t)

class CreateSummaryWriterOp final : public user_op::OpKernel {
 public:
  CreateSummaryWriterOp() = default;
  ~CreateSummaryWriterOp() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const std::string& logdir = ctx->Attr<std::string>("logdir");
    Global<EventsWriter>::Get()->Init(logdir);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return true; }
};

REGISTER_USER_KERNEL("create_summary_writer")
    .SetCreateFn<CreateSummaryWriterOp>()
    .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) { return true; });

template<typename T>
class WriteHistogramOp final : public user_op::OpKernel {
 public:
  WriteHistogramOp() = default;
  ~WriteHistogramOp() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* step = ctx->Tensor4ArgNameAndIndex("step", 0);
    const user_op::Tensor* tag = ctx->Tensor4ArgNameAndIndex("tag", 0);
    const user_op::Tensor* value = ctx->Tensor4ArgNameAndIndex("in", 0);
    int64_t* istep = const_cast<int64_t*>(step->dptr<int64_t>());
    CHECK_NOTNULL(istep);
    int8_t* ctag = const_cast<int8_t*>(tag->dptr<int8_t>());
    CHECK_NOTNULL(ctag);
    WriteHistogram<T>(static_cast<float>(istep[0]), *value, reinterpret_cast<char*>(ctag));
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return true; }
};

#define REGISTER_HISTOGRAM_USER_KERNEL(dtype)                                         \
  REGISTER_USER_KERNEL("write_histogram")                                             \
      .SetCreateFn<WriteHistogramOp<dtype>>()                                         \
      .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {                    \
        const user_op::TensorDesc* in_desc = ctx.TensorDesc4ArgNameAndIndex("in", 0); \
        return in_desc->data_type() == GetDataType<dtype>::value;                     \
      });

REGISTER_HISTOGRAM_USER_KERNEL(double)
REGISTER_HISTOGRAM_USER_KERNEL(float)
REGISTER_HISTOGRAM_USER_KERNEL(int64_t)
REGISTER_HISTOGRAM_USER_KERNEL(int32_t)
REGISTER_HISTOGRAM_USER_KERNEL(int8_t)

class WriteTextOp final : public user_op::OpKernel {
 public:
  WriteTextOp() = default;
  ~WriteTextOp() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* step = ctx->Tensor4ArgNameAndIndex("step", 0);
    const user_op::Tensor* tag = ctx->Tensor4ArgNameAndIndex("tag", 0);
    const user_op::Tensor* value = ctx->Tensor4ArgNameAndIndex("in", 0);
    int64_t* istep = const_cast<int64_t*>(step->dptr<int64_t>());
    CHECK_NOTNULL(istep);
    int8_t* ctag = const_cast<int8_t*>(tag->dptr<int8_t>());
    CHECK_NOTNULL(ctag);
    WriteText(static_cast<float>(istep[0]), *value, reinterpret_cast<char*>(ctag));
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return true; }
};

REGISTER_USER_KERNEL("write_text")
    .SetCreateFn<WriteTextOp>()
    .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) { return true; });

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
    Event e;
    Summary sum;
    std::string str = reinterpret_cast<char*>(cvalue);
    TxtString2PbMessage(str, &sum);
    e.set_step(istep[0]);
    e.set_wall_time(envtime::GetWallTime());
    *e.mutable_summary() = sum;
    Global<EventsWriter>::Get()->WriteEvent(e);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return true; }
};

REGISTER_USER_KERNEL("write_pb")
    .SetCreateFn<WritePb>()
    .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) { return true; });

class WriteImageOp final : public user_op::OpKernel {
 public:
  WriteImageOp() = default;
  ~WriteImageOp() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* step = ctx->Tensor4ArgNameAndIndex("step", 0);
    const user_op::Tensor* tag = ctx->Tensor4ArgNameAndIndex("tag", 0);
    const user_op::Tensor* value = ctx->Tensor4ArgNameAndIndex("in", 0);
    // const user_op::Tensor* bad_color = ctx->Tensor4ArgNameAndIndex("bad_color", 0);
    // const user_op::Tensor* max_images = ctx->Tensor4ArgNameAndIndex("max_images", 0);
    int64_t* istep = const_cast<int64_t*>(step->dptr<int64_t>());
    CHECK_NOTNULL(istep);
    char* ctag = const_cast<char*>(tag->dptr<char>());
    CHECK_NOTNULL(ctag);
    // int64_t* imax_images = const_cast<int64_t*>(max_images->dptr<int64_t>());
    // CHECK_NOTNULL(imax_images);
    // WriteImage(static_cast<int64_t>(istep[0]), value, ctag, static_cast<int64_t>(imax_images[0]),
    //           bad_color);
    WriteImage(static_cast<int64_t>(istep[0]), value, ctag);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return true; }
};

REGISTER_USER_KERNEL("write_image")
    .SetCreateFn<WriteImageOp>()
    .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) { return true; });

}  // namespace
}  // namespace oneflow