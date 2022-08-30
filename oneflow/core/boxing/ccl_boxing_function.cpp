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
#include "oneflow/core/framework/id_util.h"
#include "oneflow/core/framework/nd_sbp.h"
#include "oneflow/core/job/nd_sbp_util.h"
#include "oneflow/core/boxing/eager_boxing_interpreter.h"
#include "oneflow/core/common/decorator.h"
#include "oneflow/core/functional/functional.h"
#include "oneflow/core/framework/user_op_registry_manager.h"

namespace oneflow {

namespace {

class EagerBoxingKernelRegContext final : public user_op::KernelRegContext {
 public:
  explicit EagerBoxingKernelRegContext(DeviceType device_type) : device_type_(device_type) {}
  ~EagerBoxingKernelRegContext() = default;

  DeviceType device_type() const override { return device_type_; }
  const ParallelContext& parallel_ctx() const override { PRINT_BUG_PROMPT_AND_ABORT(); }
  const user_op::TensorDesc* TensorDesc4ArgNameAndIndex(const std::string& arg_name,
                                                        int32_t index) const override {
    PRINT_BUG_PROMPT_AND_ABORT();
  }
  const std::vector<std::pair<std::string, int32_t>>& inputs() const override {
    PRINT_BUG_PROMPT_AND_ABORT();
  }
  const std::vector<std::pair<std::string, int32_t>>& outputs() const override {
    PRINT_BUG_PROMPT_AND_ABORT();
  }

  const user_op::UserOpConfWrapper& user_op_conf() const override { PRINT_BUG_PROMPT_AND_ABORT(); }

  const std::shared_ptr<const user_op::AttrVal>& Attr4Name(
      const std::string& attr_name) const override {
    PRINT_BUG_PROMPT_AND_ABORT();
  }

 private:
  DeviceType device_type_;
};

Maybe<bool> RawCheckCclKernelRegistered(const std::string& op_type_name, DeviceType device_type) {
  EagerBoxingKernelRegContext reg_ctx(device_type);
  return user_op::UserOpRegistryMgr::Get().IsOpKernelRegistered(op_type_name, reg_ctx);
}

static constexpr auto* CheckCclKernelRegistered =
    DECORATE(&RawCheckCclKernelRegistered, ThreadLocalCachedCopiable);

Maybe<void> RawCheckCclP2B(Symbol<PlacedNdSbp> in, Symbol<PlacedNdSbp> out,
                           const Shape& logical_shape) {
  // NOLINTBEGIN(maybe-need-error-msg)
  CHECK_EQ_OR_RETURN(in->nd_sbp()->sbp_parallel_size(), 1);
  CHECK_EQ_OR_RETURN(out->nd_sbp()->sbp_parallel_size(), 1);

  CHECK_OR_RETURN(NdSbpIsAllPartialSum(*in->nd_sbp()));
  CHECK_OR_RETURN(NdSbpIsAllBroadcast(*out->nd_sbp()));

  CHECK_OR_RETURN(in->placement() == out->placement());
  CHECK_OR_RETURN(                                                      // NOLINT
      JUST(CheckCclKernelRegistered("eager_ccl_all_reduce",             // NOLINT
                                    in->placement()->device_type())));  // NOLINT
  // NOLINTEND(maybe-need-error-msg)
  return Maybe<void>::Ok();
}

static constexpr auto* CheckCclP2B = DECORATE(&RawCheckCclP2B, ThreadLocalCachedCopiable);

Maybe<void> RawCheckCclP2S(Symbol<PlacedNdSbp> in, Symbol<PlacedNdSbp> out,
                           const Shape& logical_shape) {
  // NOLINTBEGIN(maybe-need-error-msg)
  CHECK_EQ_OR_RETURN(in->nd_sbp()->sbp_parallel_size(), 1);
  CHECK_EQ_OR_RETURN(out->nd_sbp()->sbp_parallel_size(), 1);
  CHECK_OR_RETURN(NdSbpIsAllPartialSum(*in->nd_sbp()));
  CHECK_OR_RETURN(NdSbpIsAllSplit(*out->nd_sbp(), 0));

  CHECK_GT_OR_RETURN(logical_shape.NumAxes(), 0);
  CHECK_OR_RETURN(logical_shape.At(0) % in->placement()->parallel_num() == 0);

  CHECK_OR_RETURN(in->placement() == out->placement());
  CHECK_OR_RETURN(                                                      // NOLINT
      JUST(CheckCclKernelRegistered("eager_ccl_reduce_scatter",         // NOLINT
                                    in->placement()->device_type())));  // NOLINT
  // NOLINTEND(maybe-need-error-msg)
  return Maybe<void>::Ok();
}

static constexpr auto* CheckCclP2S = DECORATE(&RawCheckCclP2S, ThreadLocalCachedCopiable);

Maybe<void> RawCheckCclS2B(Symbol<PlacedNdSbp> in, Symbol<PlacedNdSbp> out,
                           const Shape& logical_shape) {
  // NOLINTBEGIN(maybe-need-error-msg)
  CHECK_EQ_OR_RETURN(in->nd_sbp()->sbp_parallel_size(), 1);
  CHECK_EQ_OR_RETURN(out->nd_sbp()->sbp_parallel_size(), 1);

  CHECK_OR_RETURN(NdSbpIsAllSplit(*in->nd_sbp(), 0));
  CHECK_OR_RETURN(NdSbpIsAllBroadcast(*out->nd_sbp()));

  CHECK_GT_OR_RETURN(logical_shape.NumAxes(), 0);
  CHECK_OR_RETURN(logical_shape.At(0) % in->placement()->parallel_num() == 0);

  CHECK_OR_RETURN(in->placement() == out->placement());
  CHECK_OR_RETURN(                                                      // NOLINT
      JUST(CheckCclKernelRegistered("eager_ccl_all_gather",             // NOLINT
                                    in->placement()->device_type())));  // NOLINT
  // NOLINTEND(maybe-need-error-msg)
  return Maybe<void>::Ok();
}

static constexpr auto* CheckCclS2B = DECORATE(&RawCheckCclS2B, ThreadLocalCachedCopiable);

Maybe<void> RawCheckCclS2S(Symbol<PlacedNdSbp> in, Symbol<PlacedNdSbp> out,
                           const Shape& logical_shape) {
  // NOLINTBEGIN(maybe-need-error-msg)
  CHECK_EQ_OR_RETURN(in->nd_sbp()->sbp_parallel_size(), 1);
  CHECK_EQ_OR_RETURN(out->nd_sbp()->sbp_parallel_size(), 1);

  CHECK_OR_RETURN(in->nd_sbp()->sbp_parallel(0).has_split_parallel());
  CHECK_OR_RETURN(out->nd_sbp()->sbp_parallel(0).has_split_parallel());
  CHECK_NE_OR_RETURN(in->nd_sbp()->sbp_parallel(0).split_parallel().axis(),
                     out->nd_sbp()->sbp_parallel(0).split_parallel().axis());

  int64_t in_split_axis = in->nd_sbp()->sbp_parallel(0).split_parallel().axis();
  int64_t out_split_axis = out->nd_sbp()->sbp_parallel(0).split_parallel().axis();
  CHECK_GT_OR_RETURN(logical_shape.NumAxes(), in_split_axis);
  CHECK_GT_OR_RETURN(logical_shape.NumAxes(), out_split_axis);
  CHECK_OR_RETURN(logical_shape.At(in_split_axis) % in->placement()->parallel_num() == 0);
  CHECK_OR_RETURN(logical_shape.At(out_split_axis) % in->placement()->parallel_num() == 0);

  CHECK_OR_RETURN(in->placement() == out->placement());
  CHECK_OR_RETURN(in->placement()->device_type() == DeviceType::kCPU
                  || in->placement()->device_type() == DeviceType::kCUDA);
  // NOLINTEND(maybe-need-error-msg)
  return Maybe<void>::Ok();
}

static constexpr auto* CheckCclS2S = DECORATE(&RawCheckCclS2S, ThreadLocalCachedCopiable);

}  // namespace

Maybe<one::Tensor> CclP2B(const std::shared_ptr<one::Tensor>& tensor, Symbol<PlacedNdSbp> in,
                          Symbol<PlacedNdSbp> out) {
  const auto& tensor_nd_sbp = JUST(tensor->nd_sbp());
  CHECK_OR_RETURN(tensor_nd_sbp == in->nd_sbp())
      << Error::RuntimeError() << "The sbp of input tensor (" << NdSbpToString(tensor_nd_sbp)
      << ") must match the input sbp (" << NdSbpToString(in->nd_sbp()) << ")";
  const auto& tensor_placement = JUST(tensor->parallel_desc());
  CHECK_OR_RETURN(tensor_placement == in->placement())
      << Error::RuntimeError() << "The placement of input tensor ("
      << *JUST(PlacementToString(tensor_placement)) << ") must match the input placement ("
      << *JUST(PlacementToString(in->placement())) << ")";
  return JUST(one::functional::GlobalAllReduce(tensor));
}

Maybe<one::Tensor> CclP2S(const std::shared_ptr<one::Tensor>& tensor, Symbol<PlacedNdSbp> in,
                          Symbol<PlacedNdSbp> out) {
  const auto& tensor_nd_sbp = JUST(tensor->nd_sbp());
  CHECK_OR_RETURN(tensor_nd_sbp == in->nd_sbp())
      << Error::RuntimeError() << "The sbp of input tensor (" << NdSbpToString(tensor_nd_sbp)
      << ") must match the input sbp (" << NdSbpToString(in->nd_sbp()) << ")";
  const auto& tensor_placement = JUST(tensor->parallel_desc());
  CHECK_OR_RETURN(tensor_placement == in->placement())
      << Error::RuntimeError() << "The placement of input tensor ("
      << *JUST(PlacementToString(tensor_placement)) << ") must match the input placement ("
      << *JUST(PlacementToString(in->placement())) << ")";

  return JUST(one::functional::GlobalReduceScatter(tensor, "sum"));
}

Maybe<one::Tensor> CclS2B(const std::shared_ptr<one::Tensor>& tensor, Symbol<PlacedNdSbp> in,
                          Symbol<PlacedNdSbp> out) {
  const auto& tensor_nd_sbp = JUST(tensor->nd_sbp());
  CHECK_OR_RETURN(tensor_nd_sbp == in->nd_sbp())
      << Error::RuntimeError() << "The sbp of input tensor (" << NdSbpToString(tensor_nd_sbp)
      << ") must match the input sbp (" << NdSbpToString(in->nd_sbp()) << ")";
  const auto& tensor_placement = JUST(tensor->parallel_desc());
  CHECK_OR_RETURN(tensor_placement == in->placement())
      << Error::RuntimeError() << "The placement of input tensor ("
      << *JUST(PlacementToString(tensor_placement)) << ") must match the input placement ("
      << *JUST(PlacementToString(in->placement())) << ")";
  return JUST(one::functional::GlobalAllGather(tensor));
}

Maybe<one::Tensor> CclS2S(const std::shared_ptr<one::Tensor>& tensor, Symbol<PlacedNdSbp> in,
                          Symbol<PlacedNdSbp> out) {
  const auto& tensor_nd_sbp = JUST(tensor->nd_sbp());
  CHECK_OR_RETURN(tensor_nd_sbp == in->nd_sbp())
      << Error::RuntimeError() << "The sbp of input tensor (" << NdSbpToString(tensor_nd_sbp)
      << ") must match the input sbp (" << NdSbpToString(in->nd_sbp()) << ")";
  const auto& tensor_placement = JUST(tensor->parallel_desc());
  CHECK_OR_RETURN(tensor_placement == in->placement())
      << Error::RuntimeError() << "The placement of input tensor ("
      << *JUST(PlacementToString(tensor_placement)) << ") must match the input placement ("
      << *JUST(PlacementToString(in->placement())) << ")";
  return JUST(one::functional::GlobalS2S(tensor, *JUST(GetSbpList(out->nd_sbp()))));
}

COMMAND(RegisterBoxingFunction("ccl-p-to-b", CheckCclP2B, &CclP2B));
COMMAND(RegisterBoxingFunction("ccl-p-to-s", CheckCclP2S, &CclP2S));
COMMAND(RegisterBoxingFunction("ccl-s-to-b", CheckCclS2B, &CclS2B));
COMMAND(RegisterBoxingFunction("ccl-s-to-s", CheckCclS2S, &CclS2S));

}  // namespace oneflow
