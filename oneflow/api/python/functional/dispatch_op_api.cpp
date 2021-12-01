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

#include "oneflow/core/common/scalar.h"
#include "oneflow/core/framework/nd_sbp.h"
#include "oneflow/core/framework/op_interp_ctx.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/core/functional/functional.h"
#include "oneflow/core/functional/function_library.h"

namespace oneflow {
namespace one {
namespace functional {

namespace impl {

ONEFLOW_FUNCTION_LIBRARY(m) {
  m.add_functor(
      "DispatchFeedInput",
      [](const std::shared_ptr<OpExpr>& op, const std::shared_ptr<Tensor>& input) -> Maybe<Tensor> {
        auto ctx = std::make_shared<FeedInputOpInterpCtx>();
        return OpInterpUtil::Dispatch<Tensor>(*op, {input}, ctx);
      });
  m.add_functor(
      "DispatchFetchOutput",
      [](const std::shared_ptr<OpExpr>& op, const std::shared_ptr<Tensor>& input) -> Maybe<Tensor> {
        auto ctx = std::make_shared<FetchOutputOpInterpCtx>();
        return OpInterpUtil::Dispatch<Tensor>(*op, {input}, ctx);
      });
  m.add_functor("DispatchFeedVariable",
                [](const std::shared_ptr<OpExpr>& op, const std::shared_ptr<Tensor>& input,
                   const Scalar& l2) -> Maybe<Tensor> {
                  auto ctx = std::make_shared<FeedVariableOpInterpCtx>();
                  ctx->_l2 = JUST(l2.As<double>());
                  return OpInterpUtil::Dispatch<Tensor>(*op, {input}, ctx);
                });
  m.add_functor(
      "DispatchOfrecordReader",
      [](const std::shared_ptr<OpExpr>& op, const std::string& data_dir, int32_t data_part_num,
         const std::string& part_name_prefix, int32_t part_name_suffix_length, int32_t batch_size,
         int32_t shuffle_buffer_size, bool random_shuffle, bool shuffle_after_epoch, int64_t seed,
         const Optional<Symbol<Device>>& device) -> Maybe<Tensor> {
        auto ctx = std::make_shared<OFRecordReaderOpInterpCtx>();
        ctx->data_dir = data_dir;
        ctx->data_part_num = data_part_num;
        ctx->part_name_prefix = part_name_prefix;
        ctx->part_name_suffix_length = part_name_suffix_length;
        ctx->batch_size = batch_size;
        ctx->shuffle_buffer_size = shuffle_buffer_size;
        ctx->random_shuffle = random_shuffle;
        ctx->shuffle_after_epoch = shuffle_after_epoch;
        ctx->seed = seed;
        ctx->device = device;
        return OpInterpUtil::Dispatch<Tensor>(*op, {}, ctx);
      });
  m.add_functor(
      "DispatchOfrecordReader",
      [](const std::shared_ptr<OpExpr>& op, const std::string& data_dir, int32_t data_part_num,
         const std::string& part_name_prefix, int32_t part_name_suffix_length, int32_t batch_size,
         int32_t shuffle_buffer_size, bool random_shuffle, bool shuffle_after_epoch, int64_t seed,
         const Symbol<ParallelDesc>& placement,
         const std::vector<Symbol<cfg::SbpParallel>>& sbp_tuple) -> Maybe<Tensor> {
        auto ctx = std::make_shared<OFRecordReaderOpInterpCtx>();
        ctx->data_dir = data_dir;
        ctx->data_part_num = data_part_num;
        ctx->part_name_prefix = part_name_prefix;
        ctx->part_name_suffix_length = part_name_suffix_length;
        ctx->batch_size = batch_size;
        ctx->shuffle_buffer_size = shuffle_buffer_size;
        ctx->random_shuffle = random_shuffle;
        ctx->shuffle_after_epoch = shuffle_after_epoch;
        ctx->seed = seed;
        ctx->parallel_desc = placement;
        ctx->nd_sbp = JUST(GetNdSbp(sbp_tuple));
        return OpInterpUtil::Dispatch<Tensor>(*op, {}, ctx);
      });
}

}  // namespace impl

}  // namespace functional
}  // namespace one
}  // namespace oneflow
