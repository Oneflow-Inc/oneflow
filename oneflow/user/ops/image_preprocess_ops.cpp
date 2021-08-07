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
#include "oneflow/core/job/sbp_parallel.h"
#include "oneflow/user/image/image_util.h"

namespace oneflow {

REGISTER_NO_GRAD_CPU_ONLY_USER_OP("crop_mirror_normalize_from_tensorbuffer")
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
      const user_op::TensorDesc& in_tensor = ctx->InputTensorDesc("in", 0);
      bool has_mirror = ctx->has_input("mirror", 0);
      if (has_mirror) {
        const user_op::TensorDesc& mirror_tensor = ctx->InputTensorDesc("mirror", 0);
        CHECK_OR_RETURN(mirror_tensor.shape().NumAxes() == 1
                        && in_tensor.shape().At(0) == mirror_tensor.shape().At(0));
      }
      user_op::TensorDesc* out_tensor = ctx->OutputTensorDesc("out", 0);
      int64_t N = in_tensor.shape().At(0);
      int64_t H = ctx->Attr<int64_t>("crop_h");
      int64_t W = ctx->Attr<int64_t>("crop_w");
      std::string color_space = ctx->Attr<std::string>("color_space");
      int64_t C = ImageUtil::IsColor(color_space) ? 3 : 1;

      CHECK_OR_RETURN(H != 0 && W != 0);
      CHECK_OR_RETURN(in_tensor.shape().NumAxes() == 1);
      std::string output_layout = ctx->Attr<std::string>("output_layout");
      if (output_layout == "NCHW") {
        *out_tensor->mut_shape() = Shape({N, C, H, W});
      } else if (output_layout == "NHWC") {
        *out_tensor->mut_shape() = Shape({N, H, W, C});
      } else {
        return Error::CheckFailedError()
               << "output_layout: " << output_layout << " is not supported";
      }
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      ctx->NewBuilder().Split(ctx->inputs(), 0).Split(ctx->outputs(), 0).Build();
      return Maybe<void>::Ok();
    })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc& in_tensor = ctx->InputTensorDesc("in", 0);
      CHECK_EQ_OR_RETURN(in_tensor.data_type(), DataType::kTensorBuffer);
      bool has_mirror = ctx->has_input("mirror", 0);
      if (has_mirror) {
        const user_op::TensorDesc& mirror_tensor = ctx->InputTensorDesc("mirror", 0);
        CHECK_EQ_OR_RETURN(mirror_tensor.data_type(), DataType::kInt8);
      }

      user_op::TensorDesc* out_tensor = ctx->OutputTensorDesc("out", 0);
      DataType output_dtype = ctx->Attr<DataType>("output_dtype");
      CHECK_EQ_OR_RETURN(output_dtype,
                         DataType::kFloat);  // only support float now; for float16 in future
      *out_tensor->mut_data_type() = output_dtype;

      return Maybe<void>::Ok();
    });

REGISTER_NO_GRAD_USER_OP("crop_mirror_normalize_from_uint8")
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
      const user_op::TensorDesc& in_tensor = ctx->InputTensorDesc("in", 0);
      bool has_mirror = ctx->has_input("mirror", 0);
      if (has_mirror) {
        const user_op::TensorDesc& mirror_tensor = ctx->InputTensorDesc("mirror", 0);
        CHECK_OR_RETURN(mirror_tensor.shape().NumAxes() == 1
                        && in_tensor.shape().At(0) == mirror_tensor.shape().At(0));
      }
      user_op::TensorDesc* out_tensor = ctx->OutputTensorDesc("out", 0);
      int64_t N = in_tensor.shape().At(0);
      int64_t H = ctx->Attr<int64_t>("crop_h");
      int64_t W = ctx->Attr<int64_t>("crop_w");
      std::string color_space = ctx->Attr<std::string>("color_space");
      int64_t C = ImageUtil::IsColor(color_space) ? 3 : 1;
      CHECK_EQ_OR_RETURN(in_tensor.shape().NumAxes(), 4);  // {N, H, W, C}
      CHECK_EQ_OR_RETURN(in_tensor.shape().At(3), C);
      if (H == 0 || W == 0) {
        H = in_tensor.shape().At(1);
        W = in_tensor.shape().At(2);
      } else {
        H = std::min(H, in_tensor.shape().At(1));
        W = std::min(W, in_tensor.shape().At(2));
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

      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      ctx->NewBuilder().Split(ctx->inputs(), 0).Split(ctx->outputs(), 0).Build();
      return Maybe<void>::Ok();
    })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc& in_tensor = ctx->InputTensorDesc("in", 0);
      CHECK_EQ_OR_RETURN(in_tensor.data_type(), DataType::kUInt8);
      bool has_mirror = ctx->has_input("mirror", 0);
      if (has_mirror) {
        const user_op::TensorDesc& mirror_tensor = ctx->InputTensorDesc("mirror", 0);
        CHECK_EQ_OR_RETURN(mirror_tensor.data_type(), DataType::kInt8);
      }
      user_op::TensorDesc* out_tensor = ctx->OutputTensorDesc("out", 0);
      DataType output_dtype = ctx->Attr<DataType>("output_dtype");
      CHECK_EQ_OR_RETURN(output_dtype,
                         DataType::kFloat);  // only support float now; for float16 in future
      *out_tensor->mut_data_type() = output_dtype;
      return Maybe<void>::Ok();
    });

REGISTER_NO_GRAD_CPU_ONLY_USER_OP("coin_flip")
    .Output("out")
    .Attr<float>("probability", 0.5)
    .Attr<int64_t>("batch_size")
    .Attr<int64_t>("seed", -1)
    .Attr<bool>("has_seed", false)
    .Attr<std::vector<std::string>>("parallel_distribution")
    .SetLogicalTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      user_op::TensorDesc* out_tensor = ctx->OutputTensorDesc("out", 0);
      int64_t batch_size = ctx->Attr<int64_t>("batch_size");
      *out_tensor->mut_shape() = Shape({batch_size});
      return Maybe<void>::Ok();
    })
    .SetPhysicalTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      user_op::TensorDesc* out_tensor = ctx->OutputTensorDesc("out", 0);
      int64_t batch_size = ctx->Attr<int64_t>("batch_size");
      const ParallelContext& parallel_ctx = ctx->parallel_ctx();
      const cfg::SbpParallel& out_sbp = ctx->SbpParallel4ArgNameAndIndex("out", 0);
      if (parallel_ctx.parallel_num() > 1 && out_sbp.has_split_parallel()) {
        BalancedSplitter bs(batch_size, parallel_ctx.parallel_num());
        *out_tensor->mut_shape() = Shape({bs.At(parallel_ctx.parallel_id()).size()});
      } else {
        *out_tensor->mut_shape() = Shape({batch_size});
      }
      return Maybe<void>::Ok();
    })
    .SetParallelDistributionInferFn([](user_op::InferParallelDistributionFnContext* ctx)
                                        -> Maybe<void> {
      const Shape& hierarchy = ctx->parallel_hierarchy();
      cfg::ParallelDistribution* output_dist = ctx->ParallelDistribution4ArgNameAndIndex("out", 0);
      // the input may be produced by tick which should be broadcast parallel dist
      std::vector<cfg::ParallelDistribution*> inputs_dist;
      for (const auto& arg_pair : ctx->inputs()) {
        inputs_dist.emplace_back(
            ctx->ParallelDistribution4ArgNameAndIndex(arg_pair.first, arg_pair.second));
      }
      const auto& dist_conf =
          ctx->user_op_conf().attr<std::vector<std::string>>("parallel_distribution");
      if (dist_conf.size() == 0) {
        FOR_RANGE(int, i, 0, hierarchy.NumAxes()) {
          output_dist->add_sbp_parallel()->mutable_split_parallel()->set_axis(0);
          for (auto* input_dist : inputs_dist) {
            input_dist->add_sbp_parallel()->mutable_broadcast_parallel();
          }
        }
      } else {
        CHECK_EQ_OR_RETURN(dist_conf.size(), hierarchy.NumAxes());
        for (const std::string& sbp_str : dist_conf) {
          cfg::SbpParallel sbp_parallel;
          CHECK_OR_RETURN(ParseSbpParallelFromString(sbp_str, &sbp_parallel));
          CHECK_OR_RETURN(
              (sbp_parallel.has_split_parallel() && sbp_parallel.split_parallel().axis() == 0)
              || sbp_parallel.has_broadcast_parallel());
          *output_dist->add_sbp_parallel() = sbp_parallel;
          for (auto* input_dist : inputs_dist) {
            input_dist->add_sbp_parallel()->mutable_broadcast_parallel();
          }
        }
      }
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      ctx->NewBuilder().Split(user_op::OpArg("out", 0), 0).Build();
      return Maybe<void>::Ok();
    })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      user_op::TensorDesc* out_tensor = ctx->OutputTensorDesc("out", 0);
      *out_tensor->mut_data_type() = DataType::kInt8;
      return Maybe<void>::Ok();
    });

REGISTER_NO_GRAD_CPU_ONLY_USER_OP("image_random_crop")
    .Input("in")
    .Output("out")
    .Attr<int32_t>("num_attempts", 10)
    .Attr<int64_t>("seed", -1)
    .Attr<bool>("has_seed", false)
    .Attr<std::vector<float>>("random_area", {0.08, 1.0})
    .Attr<std::vector<float>>("random_aspect_ratio", {0.75, 1.333333})
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc& in_tensor = ctx->InputTensorDesc("in", 0);
      user_op::TensorDesc* out_tensor = ctx->OutputTensorDesc("out", 0);
      *out_tensor->mut_shape() = in_tensor.shape();
      *out_tensor->mut_is_dynamic() = in_tensor.is_dynamic();
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn(user_op::GetSbpFnUtil::SplitForEachAxis)
    .SetInputArgModifyFn([](user_op::GetInputArgModifier GetInputArgModifierFn,
                            const user_op::UserOpConfWrapper&) -> Maybe<void> {
      user_op::InputArgModifier* in_modifier = GetInputArgModifierFn("in", 0);
      CHECK_NOTNULL_OR_RETURN(in_modifier);
      in_modifier->set_requires_grad(false);
      return Maybe<void>::Ok();
    })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc& in_tensor = ctx->InputTensorDesc("in", 0);
      CHECK_OR_RETURN(in_tensor.data_type() == DataType::kTensorBuffer);
      *ctx->OutputDType("out", 0) = in_tensor.data_type();
      return Maybe<void>::Ok();
    });

}  // namespace oneflow
