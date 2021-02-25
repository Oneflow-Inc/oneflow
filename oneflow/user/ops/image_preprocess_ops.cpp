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
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/user/image/image_util.h"

namespace oneflow {

REGISTER_CPU_ONLY_USER_OP("crop_mirror_normalize_from_tensorbuffer")
    .Input("in")
    .OptionalInput("mirror")
    .Output("out")
    .Attr<std::string>("color_space", "BGR")
    .Attr<std::string>("output_layout", "NCHW")
    .Attr<std::vector<float>>("mean", {0.0})
    .Attr<std::vector<float>>("std", {1.0})
    .Attr<int64_t>("crop_h", 0)
    .Attr<int64_t>("crop_w", 0)
    .Attr<float>("crop_pos_x", 0.5)
    .Attr<float>("crop_pos_y", 0.5)
    .Attr<DataType>("output_dtype", DataType::kFloat)
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

      CHECK_EQ_OR_RETURN(in_tensor->data_type(), DataType::kTensorBuffer);
      CHECK_OR_RETURN(H != 0 && W != 0);
      CHECK_OR_RETURN(in_tensor->shape().NumAxes() == 1);
      std::string output_layout = ctx->Attr<std::string>("output_layout");
      if (output_layout == "NCHW") {
        *out_tensor->mut_shape() = Shape({N, C, H, W});
      } else if (output_layout == "NHWC") {
        *out_tensor->mut_shape() = Shape({N, H, W, C});
      } else {
        return Error::CheckFailedError()
               << "output_layout: " << output_layout << " is not supported";
      }
      DataType output_dtype = ctx->Attr<DataType>("output_dtype");
      CHECK_EQ_OR_RETURN(output_dtype,
                         DataType::kFloat);  // only support float now; for float16 in future
      *out_tensor->mut_data_type() = output_dtype;
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      ctx->NewBuilder().Split(ctx->inputs(), 0).Split(ctx->outputs(), 0).Build();
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP("crop_mirror_normalize_from_uint8")
    .Input("in")
    .OptionalInput("mirror")
    .Output("out")
    .Attr<std::string>("color_space", "BGR")
    .Attr<std::string>("output_layout", "NCHW")
    .Attr<std::vector<float>>("mean", {0.0})
    .Attr<std::vector<float>>("std", {1.0})
    .Attr<int64_t>("crop_h", 0)
    .Attr<int64_t>("crop_w", 0)
    .Attr<float>("crop_pos_x", 0.5)
    .Attr<float>("crop_pos_y", 0.5)
    .Attr<DataType>("output_dtype", DataType::kFloat)
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
      CHECK_EQ_OR_RETURN(in_tensor->data_type(), DataType::kUInt8);
      CHECK_EQ_OR_RETURN(in_tensor->shape().NumAxes(), 4);  // {N, H, W, C}
      CHECK_EQ_OR_RETURN(in_tensor->shape().At(3), C);
      if (H == 0 || W == 0) {
        H = in_tensor->shape().At(1);
        W = in_tensor->shape().At(2);
      } else {
        H = std::min(H, in_tensor->shape().At(1));
        W = std::min(W, in_tensor->shape().At(2));
      }
      std::string output_layout = ctx->Attr<std::string>("output_layout");
      if (output_layout == "NCHW") {
        *out_tensor->mut_shape() = Shape({N, C, H, W});
      } else if (output_layout == "NHWC") {
        *out_tensor->mut_shape() = Shape({N, H, W, C});
      } else {
        return Error::CheckFailedError()
               << "output_layout: " << output_layout << " is not supported";
      }
      DataType output_dtype = ctx->Attr<DataType>("output_dtype");
      CHECK_EQ_OR_RETURN(output_dtype,
                         DataType::kFloat);  // only support float now; for float16 in future
      *out_tensor->mut_data_type() = output_dtype;
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      ctx->NewBuilder().Split(ctx->inputs(), 0).Split(ctx->outputs(), 0).Build();
      return Maybe<void>::Ok();
    });

REGISTER_CPU_ONLY_USER_OP("coin_flip")
    .Output("out")
    .Attr<float>("probability", 0.5)
    .Attr<int64_t>("batch_size")
    .Attr<int64_t>("seed", -1)
    .Attr<bool>("has_seed", false)
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
    });

REGISTER_CPU_ONLY_USER_OP("image_random_crop")
    .Input("in")
    .Output("out")
    .Attr<int32_t>("num_attempts", 10)
    .Attr<int64_t>("seed", -1)
    .Attr<bool>("has_seed", false)
    .Attr<std::vector<float>>("random_area", {0.08, 1.0})
    .Attr<std::vector<float>>("random_aspect_ratio", {0.75, 1.333333})
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      user_op::TensorDesc* in_tensor = ctx->TensorDesc4ArgNameAndIndex("in", 0);
      user_op::TensorDesc* out_tensor = ctx->TensorDesc4ArgNameAndIndex("out", 0);
      CHECK_OR_RETURN(in_tensor->data_type() == DataType::kTensorBuffer);
      *out_tensor = *in_tensor;
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn(user_op::GetSbpFnUtil::SplitForEachAxis)
    .SetInputArgModifyFn([](user_op::GetInputArgModifier GetInputArgModifierFn,
                            const user_op::UserOpConfWrapper&) {
      user_op::InputArgModifier* in_modifier = GetInputArgModifierFn("in", 0);
      CHECK_NOTNULL(in_modifier);
      in_modifier->set_requires_grad(false);
    });

}  // namespace oneflow
