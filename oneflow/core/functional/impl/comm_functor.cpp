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

#include "oneflow/core/framework/attr_map.h"
#include "oneflow/core/framework/attr_value.h"
#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/framework/op_interpreter/eager_mirrored_op_interpreter.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/core/functional/functional.h"
#include "oneflow/core/functional/function_library.h"
#include "oneflow/core/functional/impl/common.h"
#include "oneflow/core/functional/impl/unary_functor.h"
#include "oneflow/core/functional/scalar.h"
#include "oneflow/core/ccl/ccl.h"
#include "oneflow/core/job/rank_group_scope.h"
#include "oneflow/core/rpc/include/global_process_ctx.h"
#include "oneflow/core/common/flat_shape.h"

namespace oneflow {
namespace one {
namespace functional {

namespace impl {
class BroadcastFunctor {
 public:
  BroadcastFunctor() = default;
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x) const {
    const auto& rank_group = JUST(RankGroupScope::CurrentRankGroup());
    std::string device_type_str = JUST(x->device())->type();
    CHECK_OR_RETURN(device_type_str == "cuda" || device_type_str == "cpu");
    DeviceType device_type = device_type_str == "cuda" ? DeviceType::kGPU : DeviceType::kCPU;
    const auto& parallel_desc = JUST(RankGroup::GetDefaultParallelDesc(device_type, rank_group));
    return one::Broadcast(x, parallel_desc);
  }
};

class AllReduceFunctor {
 public:
  AllReduceFunctor() = default;
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x) const {
    {
      const auto& device = JUST(x->device());
      CHECK_EQ_OR_RETURN(JUST(device->of_type()), "gpu");
      CHECK_EQ_OR_RETURN(device->device_id(), GlobalProcessCtx::LocalRank());
    }
    static thread_local std::unordered_map<Symbol<RankGroup>, std::shared_ptr<OpExpr>>
        rank_group2op_expr;
    const auto& rank_group = JUST(RankGroupScope::CurrentRankGroup());
    auto iter = rank_group2op_expr.find(rank_group);
    std::shared_ptr<OpExpr> op_expr;
    if (iter == rank_group2op_expr.end()) {
      ParallelConf parallel_conf;
      parallel_conf.set_device_tag("gpu");
      JUST(rank_group->ForEachRank([&parallel_conf](int64_t rank) -> Maybe<void> {
        parallel_conf.add_device_name("@" + std::to_string(rank) + ":"
                                      + std::to_string(GlobalProcessCtx::LocalRank(rank)));
        return Maybe<void>::Ok();
      }));

      op_expr = JUST(one::OpBuilder("eager_nccl_all_reduce")
                         .Input("in")
                         .Output("out")
                         .Attr("parallel_conf", PbMessage2TxtString(parallel_conf))
                         .Build());
      rank_group2op_expr[rank_group] = op_expr;
    } else {
      op_expr = iter->second;
    }
    if (const auto& static_zeros_tensor = std::dynamic_pointer_cast<StaticZerosTensor>(x)) {
      return OpInterpUtil::Dispatch<Tensor>(*op_expr,
                                            {JUST(static_zeros_tensor->AsMirroredTensor())}, {});
    } else {
      return OpInterpUtil::Dispatch<Tensor>(*op_expr, {x}, {});
    }
  }
};

class SendWithoutMetaFunctor {
 public:
  SendWithoutMetaFunctor() { op_expr_ = CHECK_JUST(one::OpBuilder("send").Input("in").Build()); }
  Maybe<void> operator()(const std::shared_ptr<one::Tensor>& x, int64_t dst) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr("dst_process_id", dst));
    JUST(OpInterpUtil::Dispatch<TensorTuple>(*op_expr_, {x}, attrs));
    return Maybe<void>::Ok();
  }

 private:
  std::shared_ptr<OpExpr> op_expr_;
};

class RecvWithoutMetaFunctor {
 public:
  RecvWithoutMetaFunctor() { op_expr_ = CHECK_JUST(one::OpBuilder("recv").Output("out").Build()); }
  Maybe<Tensor> operator()(int64_t src, const Shape& shape, Symbol<DType> dtype,
                           Symbol<Device> device, const Optional<one::Tensor>& out) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr("src_process_id", src));
    JUST(attrs.SetAttr("shape", shape));
    JUST(attrs.SetAttr("dtype", dtype->data_type()));

    OpExprInterpContext op_expr_interp_context(attrs, device);

    if (out.has_value()) {
      std::shared_ptr<one::Tensor> out_tensor = JUST(out.value());
      std::shared_ptr<TensorTuple> outputs = std::make_shared<TensorTuple>(1);
      outputs->at(0) = out_tensor;
      JUST(OpInterpUtil::Dispatch(*op_expr_, {}, outputs.get(), op_expr_interp_context));
      return outputs->at(0);
    }
    return OpInterpUtil::Dispatch<Tensor>(*op_expr_, {}, op_expr_interp_context);
  }

 private:
  std::shared_ptr<OpExpr> op_expr_;
};

class SendFunctor {
 public:
  SendFunctor() { op_expr_ = CHECK_JUST(one::OpBuilder("send").Input("in").Build()); }
  Maybe<void> operator()(const std::shared_ptr<one::Tensor>& x, int64_t dst) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr("dst_process_id", dst));
    std::shared_ptr<FlatShape> flat_shape = JUST(FlatShape::New(*x->shape()));
    JUST(ccl::Send<DeviceType::kCPU>(flat_shape.get(), sizeof(*flat_shape), DataType::kChar, dst,
                                     nullptr));
    DeviceType device_type = JUST(x->device())->parallel_desc_ptr()->device_type();
    JUST(ccl::Send<DeviceType::kCPU>(&device_type, sizeof(device_type), DataType::kChar, dst,
                                     nullptr));
    DataType dtype = x->dtype()->data_type();
    JUST(ccl::Send<DeviceType::kCPU>(&dtype, sizeof(dtype), DataType::kChar, dst, nullptr));
    JUST(OpInterpUtil::Dispatch<TensorTuple>(*op_expr_, {x}, attrs));
    return Maybe<void>::Ok();
  }

 private:
  std::shared_ptr<OpExpr> op_expr_;
};

class RecvFunctor {
 public:
  RecvFunctor() { op_expr_ = CHECK_JUST(one::OpBuilder("recv").Output("out").Build()); }
  Maybe<Tensor> operator()(int64_t src, const Optional<one::Tensor>& out) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr("src_process_id", src));
    FlatShape flat_shape;
    JUST(ccl::Recv<DeviceType::kCPU>(&flat_shape, sizeof(flat_shape), DataType::kChar, src,
                                     nullptr));
    DeviceType device_type;
    JUST(ccl::Recv<DeviceType::kCPU>(&device_type, sizeof(device_type), DataType::kChar, src,
                                     nullptr));
    Symbol<Device> device =
        JUST(Device::New(Device::Type4DeviceTag(*JUST(DeviceTag4DeviceType(device_type)))));
    DataType dtype = DataType::kInvalidDataType;
    JUST(ccl::Recv<DeviceType::kCPU>(&dtype, sizeof(dtype), DataType::kChar, src, nullptr));
    JUST(attrs.SetAttr("shape", *JUST(flat_shape.ToShape())));
    JUST(attrs.SetAttr("dtype", dtype));

    OpExprInterpContext op_expr_interp_context(attrs, device);

    if (out.has_value()) {
      std::shared_ptr<one::Tensor> out_tensor = JUST(out.value());
      Symbol<Device> out_tensor_device = JUST(out_tensor->device());
      CHECK_EQ_OR_RETURN(out_tensor_device->parallel_desc_ptr()->device_type(), device_type);
      if (device_type == DeviceType::kGPU) {
        CHECK_EQ_OR_RETURN(out_tensor_device->device_id(), device->device_id());
      }
      std::shared_ptr<TensorTuple> outputs = std::make_shared<TensorTuple>(1);
      outputs->at(0) = out_tensor;
      JUST(OpInterpUtil::Dispatch(*op_expr_, {}, outputs.get(), op_expr_interp_context));
      return outputs->at(0);
    }
    return OpInterpUtil::Dispatch<Tensor>(*op_expr_, {}, op_expr_interp_context);
  }

 private:
  std::shared_ptr<OpExpr> op_expr_;
};
}  // namespace impl

ONEFLOW_FUNCTION_LIBRARY(m) {
  m.add_functor<impl::AllReduceFunctor>("AllReduce");
  m.add_functor<impl::BroadcastFunctor>("Broadcast");
  m.add_functor<impl::SendWithoutMetaFunctor>("SendWithoutMeta");
  m.add_functor<impl::RecvWithoutMetaFunctor>("RecvWithoutMeta");
  m.add_functor<impl::SendFunctor>("Send");
  m.add_functor<impl::RecvFunctor>("Recv");
};

}  // namespace functional
}  // namespace one
}  // namespace oneflow
