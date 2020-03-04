#include "oneflow/core/framework/infer_util.h"
#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/common/shape.h"
#include "oneflow/core/framework/user_op_def.pb.h"
#include "oneflow/core/operator/op_conf.pb.h"
#include "oneflow/core/framework/user_op_attr.h"
#include "oneflow/core/framework/user_op_def.h"
#include "oneflow/core/framework/user_op_conf.h"
#include "oneflow/core/framework/attr_value_accessor.h"

namespace oneflow {

namespace user_op {

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

Maybe<void> CheckAttrFnUtil::NoCheck(const UserOpDefWrapper&, const UserOpConfWrapper&) {
  return Maybe<void>::Ok();
}

size_t TmpSizeInferFnUtil::ZeroTmpSize(InferContext*) { return 0; }

}  // namespace user_op

}  // namespace oneflow
