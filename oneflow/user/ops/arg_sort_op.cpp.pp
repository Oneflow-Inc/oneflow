














#include"oneflow/core/framework/framework.h"

namespace oneflow {

static ::oneflow::user_op::UserOpRegisterTrigger<::oneflow::user_op::OpRegistry> g_register_trigger0 = ::oneflow::user_op::UserOpRegistryMgr::Get().CheckAndGetOpRegistry("arg_sort").NoGrad()
    .Input("in")
    .Output("out")
    .Attr<std::string>("direction")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->OutputShape("out", 0) = ctx->InputShape("in", 0);
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      
      
      const user_op::TensorDesc& in_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0);
      for (int64_t i = (0), __end = (in_tensor.shape().NumAxes() - 1); i < __end; ++i) {
        ctx->NewBuilder().Split(ctx->inputs(), i).Split(ctx->outputs(), i).Build();
      }
      return Maybe<void>::Ok();
    })
    .SetCheckAttrFn([](const user_op::UserOpDefWrapper& op_def,
                       const user_op::UserOpConfWrapper& op_conf) -> Maybe<void> {
      const std::string& direction = op_conf.attr<std::string>("direction");
      if (!(direction == "ASCENDING" || direction == "DESCENDING")) return Error::CheckFailedError().AddStackFrame("/home/caishenghang/oneflow-table-gen/oneflow/user/ops/arg_sort_op.cpp", 40, __FUNCTION__) << " Check failed: " << "direction == \"ASCENDING\" || direction == \"DESCENDING\"" << " ";
      return Maybe<void>::Ok();
    })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->OutputDType("out", 0) = DataType::kInt32;
      return Maybe<void>::Ok();
    });

} 
