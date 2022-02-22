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
#include "oneflow/core/boxing/eager_boxing_interpreter_mgr.h"
#include "oneflow/core/boxing/eager_boxing_logger.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/framework/placement_sbp_util.h"
#include "oneflow/core/framework/placed_nd_sbp.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/id_util.h"
#include "oneflow/core/common/decorator.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/functional/functional.h"

namespace oneflow {

namespace {

Maybe<one::OpExpr> MakeToConsistentOpExpr() {
  std::shared_ptr<one::OpExpr> op_expr =
      JUST(one::CastToConsistentOpExpr::New(*JUST(UniqueStr("cast_to_consistent"))));
  return op_expr;
}

static constexpr auto* GetLocalToConsistentOpExpr =
    DECORATE(&MakeToConsistentOpExpr, ThreadLocalCopiable);

Maybe<one::Tensor> ReinterpterConsistentTensor(const std::shared_ptr<one::Tensor>& tensor,
                                               const Shape& shape,
                                               Symbol<ParallelDesc> parallel_desc,
                                               Symbol<NdSbp> nd_sbp) {
  const auto& op = JUST(GetLocalToConsistentOpExpr());
  MutableAttrMap attrs;
  JUST(attrs.SetAttr<Shape>("shape", shape));
  JUST(attrs.SetAttr<DataType>("dtype", tensor->dtype()->data_type()));
  const auto& parallel_id = JUST(GetParallelId4CurrentProcessCtx(parallel_desc));
  std::shared_ptr<Shape> pyhsical_shape =
      JUST(GetPhysicalShape(shape, *nd_sbp, *parallel_desc, JUST(*parallel_id)));
  std::shared_ptr<one::Tensor> x = JUST(tensor->cur_rank_phy_tensor());
  if (*x->shape() != *pyhsical_shape) { x = JUST(one::functional::Reshape(x, *pyhsical_shape)); }
  return JUST(one::OpInterpUtil::Dispatch<one::Tensor>(
      *op, {x}, one::OpExprInterpContext(attrs, parallel_desc, nd_sbp)));
}

Maybe<one::Tensor> Apply1DBoxing(const std::shared_ptr<one::Tensor>& input, Symbol<NdSbp> in_nd_sbp,
                                 Symbol<NdSbp> out_nd_sbp, Symbol<ParallelDesc> in_parallel_desc,
                                 Symbol<ParallelDesc> out_parallel_desc) {
  const auto& boxing_interpreter =
      JUST(Global<EagerBoxingInterpreterManager>::Get()->GetEagerBoxingInterpreter(
          in_nd_sbp, out_nd_sbp, in_parallel_desc, out_parallel_desc, *input->shape()));
  Global<const EagerBoxingLogger>::Get()->Log(
      *JUST(boxing_interpreter->boxing_interpreter_status()),
      /* prefix */ "\t\tInternal boxing of symmetric-acyclic-nd-sbp-to-nd-sbp, ");
  return JUST(boxing_interpreter->Interpret(input, in_nd_sbp, out_nd_sbp, in_parallel_desc,
                                            out_parallel_desc));
}

}  // namespace

Maybe<void> CheckSymmetricAcyclicNdSbpBoxing(Symbol<PlacedNdSbp> in, Symbol<PlacedNdSbp> out,
                                             const Shape& logical_shape) {
  CHECK_OR_RETURN(in->placement() == out->placement());
  CHECK_OR_RETURN(in->nd_sbp() != out->nd_sbp());
  CHECK_EQ_OR_RETURN(in->nd_sbp()->sbp_parallel_size(), out->nd_sbp()->sbp_parallel_size());
  CHECK_GT_OR_RETURN(in->nd_sbp()->sbp_parallel_size(), 1);
  JUST(CheckIsNdSbpBoxingAcyclicWithDecompose(in, out, logical_shape));
  return Maybe<void>::Ok();
}

Maybe<one::Tensor> SymmetricAcyclicNdSbpBoxing(const std::shared_ptr<one::Tensor>& input,
                                               Symbol<PlacedNdSbp> in, Symbol<PlacedNdSbp> out) {
  const auto& out_nd_sbp = out->nd_sbp();
  const auto& in_parallel_desc = in->placement();
  const auto& out_parallel_desc = out->placement();
  CHECK_OR_RETURN(in_parallel_desc == out_parallel_desc);
  const auto& tensor_meta = JUST(input->consistent_tensor_meta());
  const auto& naive_transformations =
      JUST(DecomposeIntoNaiveTransformations(tensor_meta, out_nd_sbp));
  std::shared_ptr<one::Tensor> tensor = input;
  for (const auto& naive_transformation : *naive_transformations) {
    const auto& sub_tensor_meta = naive_transformation.consistent_tensor_meta;
    tensor = JUST(ReinterpterConsistentTensor(tensor, sub_tensor_meta->shape(),
                                              sub_tensor_meta->parallel_desc(),
                                              sub_tensor_meta->nd_sbp()));
    tensor =
        JUST(Apply1DBoxing(tensor, sub_tensor_meta->nd_sbp(), naive_transformation.dst_nd_sbp,
                           sub_tensor_meta->parallel_desc(), sub_tensor_meta->parallel_desc()));
  }
  return JUST(ReinterpterConsistentTensor(tensor, *input->shape(), out_parallel_desc, out_nd_sbp));
}

COMMAND(RegisterBoxingFunction("symmetric-acyclic-nd-sbp-to-nd-sbp",
                               CheckSymmetricAcyclicNdSbpBoxing, &SymmetricAcyclicNdSbpBoxing));

}  // namespace oneflow
