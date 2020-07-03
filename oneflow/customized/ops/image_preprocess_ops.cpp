#include "oneflow/core/framework/framework.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/customized/image/image_util.h"

namespace oneflow {

REGISTER_CPU_ONLY_USER_OP("image_resize")
    .Input("in")
    .Output("out")
    .Attr<std::string>("color_space", UserOpAttrType::kAtString, "BGR")
    .Attr<std::string>("interp_type", UserOpAttrType::kAtString, "Linear")          /*not use*/
    .Attr<std::string>("mag_filter", UserOpAttrType::kAtString, "Linear")           /*not use*/
    .Attr<std::string>("min_filter", UserOpAttrType::kAtString, "Linear")           /*not use*/
    .Attr<std::vector<float>>("max_size", UserOpAttrType::kAtListFloat, {0.0, 0.0}) /*not use*/
    .Attr<int64_t>("resize_longer", UserOpAttrType::kAtInt64, 0)                    /*not use*/
    .Attr<int64_t>("resize_shorter", UserOpAttrType::kAtInt64, 0)
    .Attr<int64_t>("resize_x", UserOpAttrType::kAtInt64, 0)
    .Attr<int64_t>("resize_y", UserOpAttrType::kAtInt64, 0)
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      user_op::TensorDesc* in_tensor = ctx->TensorDesc4ArgNameAndIndex("in", 0);
      user_op::TensorDesc* out_tensor = ctx->TensorDesc4ArgNameAndIndex("out", 0);
      CHECK_OR_RETURN(in_tensor->data_type() == DataType::kTensorBuffer);
      CHECK_OR_RETURN(in_tensor->shape().NumAxes() == 1 && in_tensor->shape().At(0) >= 1);
      int64_t batch_size = in_tensor->shape().At(0);
      int64_t resize_x = ctx->Attr<int64_t>("resize_x");
      int64_t resize_y = ctx->Attr<int64_t>("resize_y");
      if (resize_x != 0 && resize_y != 0) {
        // resize_x -> W
        // resize_y -> H
        // shape = {H, W, c}
        std::string color_space = ctx->Attr<std::string>("color_space");
        int64_t c = ImageUtil::IsColor(color_space) ? 3 : 1;
        *out_tensor->mut_data_type() = DataType::kUInt8;
        *out_tensor->mut_shape() = Shape({batch_size, resize_y, resize_x, c});
      } else {
        CHECK_OR_RETURN(ctx->Attr<int64_t>("resize_shorter") != 0);
        *out_tensor->mut_data_type() = DataType::kTensorBuffer;
        *out_tensor->mut_shape() = Shape({batch_size});
      }
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      ctx->NewBuilder()
          .Split(user_op::OpArg("in", 0), 0)
          .Split(user_op::OpArg("out", 0), 0)
          .Build();
      return Maybe<void>::Ok();
    })
    .SetBatchAxisInferFn([](user_op::BatchAxisContext* ctx) -> Maybe<void> {
      CHECK_EQ_OR_RETURN(ctx->BatchAxis4ArgNameAndIndex("in", 0)->value(), 0);
      ctx->BatchAxis4ArgNameAndIndex("out", 0)->set_value(0);
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
