#include "oneflow/core/framework/framework.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/customized/image/image_util.h"

namespace oneflow {

REGISTER_CPU_ONLY_USER_OP("image_resize")
    .Input("in")
    .Output("out")
    .Output("scale")
    .Attr<int64_t>("target_width", UserOpAttrType::kAtInt64, 0)
    .Attr<int64_t>("target_height", UserOpAttrType::kAtInt64, 0)
    .Attr<int64_t>("channels", UserOpAttrType::kAtInt64, 3)
    .Attr<DataType>("data_type", UserOpAttrType::kAtDataType, DataType::kUInt8)
    .Attr<std::string>("interpolation", UserOpAttrType::kAtString, "bilinear")
    .SetCheckAttrFn([](const user_op::UserOpDefWrapper& def,
                       const user_op::UserOpConfWrapper& conf) -> Maybe<void> {
      bool check_failed = false;
      std::ostringstream err;
      err << "Illegal attr value for " << conf.op_type_name() << " op, op_name: " << conf.op_name();
      int64_t target_width = conf.attr<int64_t>("target_width");
      int64_t target_height = conf.attr<int64_t>("target_height");
      if (target_width <= 0 || target_height <= 0) {
        err << ", target_width: " << target_width << ", target_height: " << target_height;
        check_failed = true;
      }
      int64_t channels = conf.attr<int64_t>("channels");
      if (channels != 1 && channels != 3) {
        err << ", channels: " << channels << " (channels can only be 1 or 3)";
        check_failed = true;
      }
      DataType data_type = conf.attr<DataType>("data_type");
      if (data_type != DataType::kUInt8 && data_type != DataType::kFloat) {
        err << ", data_type: " << data_type << " (only support kUInt8 and kFloat for now)";
        check_failed = true;
      }
      const std::string& interpolation = conf.attr<std::string>("interpolation");
      if (!CheckInterpolationValid(interpolation, err)) { check_failed = true; }
      if (check_failed) { return oneflow::Error::CheckFailed() << err.str(); }
      return Maybe<void>::Ok();
    })
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc* in_tensor = ctx->TensorDesc4ArgNameAndIndex("in", 0);
      CHECK_OR_RETURN(in_tensor->data_type() == DataType::kTensorBuffer);
      CHECK_OR_RETURN(in_tensor->shape().NumAxes() == 1 && in_tensor->shape().elem_cnt() > 0);
      int64_t batch_size = in_tensor->shape().elem_cnt();
      int64_t target_width = ctx->Attr<int64_t>("target_width");
      int64_t target_height = ctx->Attr<int64_t>("target_height");
      int64_t channels = ctx->Attr<int64_t>("channels");

      user_op::TensorDesc* out_tensor = ctx->TensorDesc4ArgNameAndIndex("out", 0);
      *out_tensor->mut_data_type() = ctx->Attr<DataType>("data_type");
      *out_tensor->mut_shape() = Shape({batch_size, target_height, target_width, channels});
      out_tensor->set_is_dynamic(in_tensor->is_dynamic());

      user_op::TensorDesc* scale_tensor = ctx->TensorDesc4ArgNameAndIndex("scale", 0);
      *scale_tensor->mut_data_type() = DataType::kFloat;
      *scale_tensor->mut_shape() = Shape({batch_size, 2});
      scale_tensor->set_is_dynamic(in_tensor->is_dynamic());

      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      ctx->NewBuilder().Split(ctx->inputs(), 0).Split(ctx->outputs(), 0).Build();
      return Maybe<void>::Ok();
    })
    .SetBatchAxisInferFn([](user_op::BatchAxisContext* ctx) -> Maybe<void> {
      CHECK_EQ_OR_RETURN(ctx->BatchAxis4ArgNameAndIndex("in", 0)->value(), 0);
      ctx->BatchAxis4ArgNameAndIndex("out", 0)->set_value(0);
      ctx->BatchAxis4ArgNameAndIndex("scale", 0)->set_value(0);
      return Maybe<void>::Ok();
    });

REGISTER_CPU_ONLY_USER_OP("crop_mirror_normalize")
    .Input("in")
    .OptionalInput("mirror")
    .Output("out")
    .Attr<std::string>("color_space", UserOpAttrType::kAtString, "BGR")
    .Attr<std::string>("output_layout", UserOpAttrType::kAtString, "NCHW")
    .Attr<std::vector<float>>("mean", UserOpAttrType::kAtListFloat, {0.0})
    .Attr<std::vector<float>>("std", UserOpAttrType::kAtListFloat, {1.0})
    .Attr<int64_t>("crop_h", UserOpAttrType::kAtInt64, 0)
    .Attr<int64_t>("crop_w", UserOpAttrType::kAtInt64, 0)
    .Attr<float>("crop_pos_x", UserOpAttrType::kAtFloat, 0.5)
    .Attr<float>("crop_pos_y", UserOpAttrType::kAtFloat, 0.5)
    .Attr<bool>("pad_output", UserOpAttrType::kAtBool, false)
    .Attr<int32_t>("output_dtype", UserOpAttrType::kAtInt32, static_cast<int32_t>(DataType::kFloat))
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      user_op::TensorDesc* in_tensor = ctx->TensorDesc4ArgNameAndIndex("in", 0);
      user_op::TensorDesc* mirror_tensor = ctx->TensorDesc4ArgNameAndIndex("mirror", 0);
      if (mirror_tensor) {
        CHECK_OR_RETURN(mirror_tensor->shape().NumAxes() == 1
                        && in_tensor->shape().At(0) == mirror_tensor->shape().At(0));
        CHECK_EQ_OR_RETURN(mirror_tensor->data_type(), DataType::kInt8);
      }
      user_op::TensorDesc* out_tensor = ctx->TensorDesc4ArgNameAndIndex("out", 0);
      int64_t N = in_tensor->shape().At(0);
      int64_t H = ctx->Attr<int64_t>("crop_h");
      int64_t W = ctx->Attr<int64_t>("crop_w");
      std::string color_space = ctx->Attr<std::string>("color_space");
      int64_t C = ImageUtil::IsColor(color_space) ? 3 : 1;
      if (in_tensor->data_type() == DataType::kUInt8) {
        CHECK_EQ_OR_RETURN(in_tensor->shape().NumAxes(), 4);  // {N, H, W, C}
        CHECK_EQ_OR_RETURN(in_tensor->shape().At(3), C);
        if (H == 0 || W == 0) {
          H = in_tensor->shape().At(1);
          W = in_tensor->shape().At(2);
        }
      } else if (in_tensor->data_type() == DataType::kTensorBuffer) {
        CHECK_OR_RETURN(H != 0 && W != 0);
        CHECK_OR_RETURN(in_tensor->shape().NumAxes() == 1);
      } else {
        return Error::CheckFailed()
               << "input Dtype: " << in_tensor->data_type() << " is not supported";
      }

      std::string output_layout = ctx->Attr<std::string>("output_layout");
      if (output_layout == "NCHW") {
        *out_tensor->mut_shape() = Shape({N, C, H, W});
      } else if (output_layout == "NHWC") {
        *out_tensor->mut_shape() = Shape({N, H, W, C});
      } else {
        return Error::CheckFailed() << "output_layout: " << output_layout << " is not supported";
      }
      DataType output_dtype = static_cast<DataType>(ctx->Attr<int32_t>("output_dtype"));
      CHECK_EQ_OR_RETURN(output_dtype,
                         DataType::kFloat);  // only support float now; for float16 in future
      *out_tensor->mut_data_type() = output_dtype;

      bool pad_output = ctx->Attr<bool>("pad_output");
      CHECK_OR_RETURN(pad_output == false);  // TODO(chengcheng)

      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      ctx->NewBuilder().Split(ctx->inputs(), 0).Split(ctx->outputs(), 0).Build();
      return Maybe<void>::Ok();
    })
    .SetBatchAxisInferFn([](user_op::BatchAxisContext* ctx) -> Maybe<void> {
      CHECK_EQ_OR_RETURN(ctx->BatchAxis4ArgNameAndIndex("in", 0)->value(), 0);
      ctx->BatchAxis4ArgNameAndIndex("out", 0)->set_value(0);
      return Maybe<void>::Ok();
    });

REGISTER_CPU_ONLY_USER_OP("coin_flip")
    .Output("out")
    .Attr<float>("probability", UserOpAttrType::kAtFloat, 0.5)
    .Attr("batch_size", UserOpAttrType::kAtInt64)
    .Attr<int64_t>("seed", UserOpAttrType::kAtInt64, -1)
    .Attr<bool>("has_seed", UserOpAttrType::kAtBool, false)
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      user_op::TensorDesc* out_tensor = ctx->TensorDesc4ArgNameAndIndex("out", 0);
      int64_t batch_size = ctx->Attr<int64_t>("batch_size");
      const ParallelContext& parallel_ctx = ctx->parallel_ctx();
      const SbpParallel& out_sbp = ctx->SbpParallel4ArgNameAndIndex("out", 0);
      if (parallel_ctx.parallel_num() > 1 && out_sbp.has_split_parallel()) {
        BalancedSplitter bs(batch_size, parallel_ctx.parallel_num());
        *out_tensor->mut_shape() = Shape({bs.At(parallel_ctx.parallel_id()).size()});
      } else {
        *out_tensor->mut_shape() = Shape({batch_size});
      }
      *out_tensor->mut_data_type() = DataType::kInt8;
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      ctx->NewBuilder().Split(user_op::OpArg("out", 0), 0).Build();
      return Maybe<void>::Ok();
    })
    .SetBatchAxisInferFn([](user_op::BatchAxisContext* ctx) -> Maybe<void> {
      ctx->BatchAxis4ArgNameAndIndex("out", 0)->set_value(0);
      return Maybe<void>::Ok();
    });

}  // namespace oneflow
