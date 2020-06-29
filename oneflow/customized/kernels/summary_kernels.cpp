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

Maybe<void> AddScalarToSummary(const float& value, const std::string& tag, Summary* s) {
  SummaryMetadata metadata;
  PatchPluginName(&metadata, kScalarPluginName);
  Summary::Value* v = s->add_value();
  v->set_tag(tag);
  *v->mutable_metadata() = metadata;
  v->set_simple_value(value);
  return Maybe<void>::Ok();
}

void WriteScalar(int64_t step, const float& value, const std::string& tag) {
  Event e;
  e.set_step(step);
  e.set_wall_time(envtime::GetWallTime());
  AddScalarToSummary(value, tag, e.mutable_summary());
  // Global<EventsWriter>::Get()->Initialize("/home/zjhushengjian/oneflow", "laoxu");
  Global<EventsWriter>::Get()->WriteEvent(e);
}

Maybe<void> AddHistogramToSummary(const user_op::Tensor& value, const std::string& tag,
                                  Summary* s) {
  SummaryMetadata metadata;
  PatchPluginName(&metadata, kHistogramPluginName);
  Summary::Value* v = s->add_value();
  v->set_tag(tag);
  *v->mutable_metadata() = metadata;
  histogram::Histogram histo;
  for (int64_t i = 0; i < value.shape().elem_cnt(); i++) {
    double double_val = value.dptr<double>()[i];
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

void WriteHistogram(const float& step, const user_op::Tensor& value, const std::string& tag) {
  Event e;
  e.set_step(step);
  e.set_wall_time(envtime::GetWallTime());
  AddHistogramToSummary(value, tag, e.mutable_summary());
  Global<EventsWriter>::Get()->WriteEvent(e);
}

void WriteText(const float& step, const user_op::Tensor& value, const std::string& tag) {
  Event e;
  e.set_step(step);
  e.set_wall_time(envtime::GetWallTime());
  AddTextToSummary(value, tag, e.mutable_summary());
  Global<EventsWriter>::Get()->WriteEvent(e);
}

class WriteScalarOp final : public user_op::OpKernel {
 public:
  WriteScalarOp() = default;
  ~WriteScalarOp() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* step = ctx->Tensor4ArgNameAndIndex("step", 0);
    const user_op::Tensor* tag = ctx->Tensor4ArgNameAndIndex("tag", 0);
    const user_op::Tensor* value = ctx->Tensor4ArgNameAndIndex("in", 0);

    double* dvalue = const_cast<double*>(value->dptr<double>());
    CHECK_NOTNULL(dvalue);
    int64_t* istep = const_cast<int64_t*>(step->dptr<int64_t>());
    CHECK_NOTNULL(istep);
    int8_t* ctag = const_cast<int8_t*>(tag->dptr<int8_t>());
    CHECK_NOTNULL(ctag);
    WriteScalar(istep[0], dvalue[0], reinterpret_cast<char*>(ctag));
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return true; }
};

REGISTER_USER_KERNEL("write_scalar")
    .SetCreateFn<WriteScalarOp>()
    .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) { return true; });

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
    WriteHistogram(static_cast<float>(istep[0]), *value, reinterpret_cast<char*>(ctag));
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return true; }
};

REGISTER_USER_KERNEL("write_histogram")
    .SetCreateFn<WriteHistogramOp>()
    .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) { return true; });

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

}  // namespace
}  // namespace oneflow