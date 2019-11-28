#include "oneflow/core/framework/op_infer_util.h"
#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/common/shape.h"
#include "oneflow/core/framework/user_op_def.pb.h"
#include "oneflow/core/operator/op_conf.pb.h"
#include "oneflow/core/framework/user_op_attr.h"

namespace oneflow {

namespace user_op {

InferContext::InferContext(Shape4ArgNameAndIndexFn shape_fn, Dtype4ArgNameAndIndexFn dtype_fn,
                           const UserOpConf* conf)
    : conf_(conf), inputs_(), outputs_(), shape_infer_fn_(shape_fn), dtype_infer_fn_(dtype_fn) {
  for (auto it = conf_->input().begin(); it != conf_->input().end(); ++it) {
    const std::string& arg_name = it->first;
    for (int i = 0; i < it->second.s_size(); ++i) {
      inputs_.emplace_back(std::make_pair(arg_name, i));
    }
  }
  for (auto it = conf_->output().begin(); it != conf_->output().end(); ++it) {
    const std::string& arg_name = it->first;
    for (int i = 0; i < it->second.s_size(); ++i) {
      outputs_.emplace_back(std::make_pair(arg_name, i));
    }
  }
}

Shape* InferContext::Shape4ArgNameAndIndex(const std::string& arg_name, int32_t index) const {
  return shape_infer_fn_(arg_name, index);
}

DataType* InferContext::Dtype4ArgNameAndIndex(const std::string& arg_name, int32_t index) const {
  return dtype_infer_fn_(arg_name, index);
}

const ArgVec& InferContext::inputs() const { return inputs_; }

const ArgVec& InferContext::outputs() const { return outputs_; }

#define BASIC_AND_MESSAGE_ATTR_TYPE_SPECIALIZATION(type, field_name)             \
  template<>                                                                     \
  type InferContext::GetAttr<type>(const std::string& attr_name) const {         \
    const UserOpAttrVal& attr_val = conf_->attr().at(attr_name);                 \
    CHECK_EQ(static_cast<int>(GetAttrType<type>::value), attr_val.value_case()); \
    return attr_val.field_name();                                                \
  }

BASIC_AND_MESSAGE_ATTR_TYPE_SPECIALIZATION(int32_t, at_int32)
BASIC_AND_MESSAGE_ATTR_TYPE_SPECIALIZATION(int64_t, at_int64)
BASIC_AND_MESSAGE_ATTR_TYPE_SPECIALIZATION(bool, at_bool)
BASIC_AND_MESSAGE_ATTR_TYPE_SPECIALIZATION(float, at_float)
BASIC_AND_MESSAGE_ATTR_TYPE_SPECIALIZATION(double, at_double)
BASIC_AND_MESSAGE_ATTR_TYPE_SPECIALIZATION(std::string, at_string)

#undef BASIC_AND_MESSAGE_ATTR_TYPE_SPECIALIZATION

#define LIST_ATTR_TYPE_SPECIALIZATION(type, field_name)                          \
  template<>                                                                     \
  type InferContext::GetAttr<type>(const std::string& attr_name) const {         \
    const UserOpAttrVal& attr_val = conf_->attr().at(attr_name);                 \
    CHECK_EQ(static_cast<int>(GetAttrType<type>::value), attr_val.value_case()); \
    type ret;                                                                    \
    SerializeListAttr2Vector(attr_val.field_name(), &ret);                       \
    return ret;                                                                  \
  }

LIST_ATTR_TYPE_SPECIALIZATION(std::vector<int32_t>, at_list_int32)
LIST_ATTR_TYPE_SPECIALIZATION(std::vector<int64_t>, at_list_int64)
LIST_ATTR_TYPE_SPECIALIZATION(std::vector<float>, at_list_float)

#undef LIST_ATTR_TYPE_SPECIALIZATION

Maybe<void> ShapeInferFnUtil::Unchanged(InferContext* ctx) {
  const Shape* shape = nullptr;
  for (size_t i = 0; i < ctx->inputs().size(); ++i) {
    const std::pair<std::string, int32_t>& input_arg = ctx->inputs().at(i);
    if (shape) {
      CHECK_EQ_OR_RETURN(*shape, *ctx->Shape4ArgNameAndIndex(input_arg.first, input_arg.second));
    } else {
      shape = ctx->Shape4ArgNameAndIndex(input_arg.first, input_arg.second);
    }
  }
  for (size_t i = 0; i < ctx->outputs().size(); ++i) {
    const std::pair<std::string, int32_t>& output_arg = ctx->outputs().at(i);
    *ctx->Shape4ArgNameAndIndex(output_arg.first, output_arg.second) = *shape;
  }
  return Maybe<void>::Ok();
}

Maybe<void> ShapeInferFnUtil::InOutCorrespond(InferContext* ctx) {
  CHECK_EQ_OR_RETURN(ctx->inputs().size(), ctx->outputs().size());
  for (size_t i = 0; i < ctx->inputs().size(); ++i) {
    const auto& input_arg = ctx->inputs().at(i);
    const auto& output_arg = ctx->outputs().at(i);
    *ctx->Shape4ArgNameAndIndex(output_arg.first, output_arg.second) =
        *ctx->Shape4ArgNameAndIndex(input_arg.first, input_arg.second);
  }
  return Maybe<void>::Ok();
}

Maybe<void> DtypeInferFnUtil::Unchanged(InferContext* ctx) {
  const DataType* dtype = nullptr;
  for (size_t i = 0; i < ctx->inputs().size(); ++i) {
    const std::pair<std::string, int32_t>& input_arg = ctx->inputs().at(i);
    if (dtype) {
      CHECK_EQ_OR_RETURN(*dtype, *ctx->Dtype4ArgNameAndIndex(input_arg.first, input_arg.second));
    } else {
      dtype = ctx->Dtype4ArgNameAndIndex(input_arg.first, input_arg.second);
    }
  }
  for (size_t i = 0; i < ctx->outputs().size(); ++i) {
    const std::pair<std::string, int32_t>& output_arg = ctx->outputs().at(i);
    *ctx->Dtype4ArgNameAndIndex(output_arg.first, output_arg.second) = *dtype;
  }
  return Maybe<void>::Ok();
}

Maybe<void> DtypeInferFnUtil::InOutCorrespond(InferContext* ctx) {
  CHECK_EQ_OR_RETURN(ctx->inputs().size(), ctx->outputs().size());
  for (size_t i = 0; i < ctx->inputs().size(); ++i) {
    const auto& input_arg = ctx->inputs().at(i);
    const auto& output_arg = ctx->outputs().at(i);
    *ctx->Dtype4ArgNameAndIndex(output_arg.first, output_arg.second) =
        *ctx->Dtype4ArgNameAndIndex(input_arg.first, input_arg.second);
  }
  return Maybe<void>::Ok();
}

Maybe<void> CheckAttrFnUtil::NoCheck(const UserOpDef&, const UserOpConf&) {
  return Maybe<void>::Ok();
}

}  // namespace user_op

}  // namespace oneflow
