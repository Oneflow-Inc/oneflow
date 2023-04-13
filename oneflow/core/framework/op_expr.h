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
#ifndef ONEFLOW_CORE_FRAMEWORK_OP_EXPR_H_
#define ONEFLOW_CORE_FRAMEWORK_OP_EXPR_H_

#include <string>
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/symbol.h"
#include "oneflow/core/common/optional.h"
#include "oneflow/core/job/sbp_parallel.h"
#include "oneflow/core/operator/op_conf.pb.h"
#include "oneflow/core/framework/attr_map.h"
#include "oneflow/core/framework/autocast.h"
#include "oneflow/core/framework/device.h"
#include "oneflow/core/framework/stream.h"
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/core/framework/user_op_conf.pb.h"
#include "oneflow/core/framework/user_op_registry.h"
#include "oneflow/core/framework/arg_tuple.h"
#include "oneflow/core/autograd/autograd_function.h"
#include "oneflow/core/job/lazy_mode.h"
#include "oneflow/core/framework/op_interpreter/dispatch_frame.h"

namespace oneflow {
namespace one {

class OpExprGradFunctionIf;
class OpExprGradClosure;

class OpExpr {
 public:
  virtual ~OpExpr() = default;
  virtual const std::string& op_type_name() const = 0;

  virtual int input_size() const = 0;
  virtual int output_size() const = 0;

  virtual Maybe<bool> IsGradDisabled() const = 0;
  virtual Maybe<bool> SupportNonContiguous() const = 0;

  virtual Maybe<OpExprGradClosure> GetOrCreateOpGradClosure() const = 0;

  virtual Maybe<autocast::AutoCastMeta> GetOrCreateAutoCastMeta() const;

 protected:
  OpExpr() = default;
};

class BuiltinOpExpr : public OpExpr {
 public:
  explicit BuiltinOpExpr(const std::string& op_name, const std::vector<std::string>& indexed_ibns,
                         const std::vector<std::string>& indexed_obns);

  virtual ~BuiltinOpExpr() = default;

  const std::string& op_name() const { return op_name_; }

  int input_size() const override { return input_arg_tuple_->size(); }
  int output_size() const override { return output_arg_tuple_->size(); }

  const std::shared_ptr<const ArgTuple>& input_arg_tuple() const { return input_arg_tuple_; }
  const std::shared_ptr<const ArgTuple>& output_arg_tuple() const { return output_arg_tuple_; }

  const std::vector<std::string>& indexed_ibns() const { return input_arg_tuple_->indexed_bns(); }
  const std::vector<std::string>& indexed_obns() const { return output_arg_tuple_->indexed_bns(); }
  const std::vector<std::pair<std::string, int32_t>>& indexed_input_pairs() const {
    return input_arg_tuple_->indexed_arg_name_and_index();
  }
  const std::vector<std::pair<std::string, int32_t>>& indexed_output_pairs() const {
    return output_arg_tuple_->indexed_arg_name_and_index();
  }

  virtual Maybe<void> BuildOpConf(OperatorConf* op_conf, const AttrMap& attrs) const = 0;

 protected:
  std::string op_name_;
  std::shared_ptr<const ArgTuple> input_arg_tuple_;
  std::shared_ptr<const ArgTuple> output_arg_tuple_;
};

class TensorMeta;

template<typename ProtoType>
class BuiltinOpExprImpl : public BuiltinOpExpr {
 public:
  static Maybe<BuiltinOpExprImpl<ProtoType>> New(const std::string& op_name, ProtoType&& op_proto,
                                                 const std::vector<std::string>& indexed_ibns,
                                                 const std::vector<std::string>& indexed_obns) {
    return std::shared_ptr<BuiltinOpExprImpl<ProtoType>>(
        new BuiltinOpExprImpl<ProtoType>(op_name, std::move(op_proto), indexed_ibns, indexed_obns));
  }

  virtual ~BuiltinOpExprImpl() = default;

  const ProtoType& proto() const { return op_proto_; }
  ProtoType* mutable_proto() { return &op_proto_; }

  const std::string& op_type_name() const override;

  Maybe<bool> IsGradDisabled() const override;

  Maybe<bool> SupportNonContiguous() const override;

  Maybe<OpExprGradClosure> GetOrCreateOpGradClosure() const override;

  Maybe<autocast::AutoCastMeta> GetOrCreateAutoCastMeta() const override;

  Maybe<void> BuildOpConf(OperatorConf* op_conf, const AttrMap& attrs) const override;

 protected:
  explicit BuiltinOpExprImpl(const std::string& op_name, ProtoType&& op_proto,
                             const std::vector<std::string>& indexed_ibns,
                             const std::vector<std::string>& indexed_obns)
      : BuiltinOpExpr(op_name, indexed_ibns, indexed_obns), op_proto_(std::move(op_proto)) {}

  ProtoType op_proto_;
  mutable std::shared_ptr<OpExprGradFunctionIf> op_grad_func_;
  mutable std::shared_ptr<autocast::AutoCastMeta> autocast_meta_;
};

class StatefulOpKernel;
class LocalTensorInferCache;
class GlobalTensorInferCache;

class UserOpExpr final : public BuiltinOpExprImpl<UserOpConf> {
 public:
  UserOpExpr() = delete;
  virtual ~UserOpExpr() = default;

  static Maybe<UserOpExpr> New(const std::string& op_name, UserOpConf&& op_proto,
                               const std::vector<std::string>& indexed_ibns,
                               const std::vector<std::string>& indexed_obns);

  const AttrMap& base_attrs() const { return base_attrs_; }

  Maybe<StatefulOpKernel> MutKernel4Stream(Symbol<Stream> stream) const;

  bool has_device_and_stream_infer_fn() const {
    return static_cast<bool>(device_and_stream_infer_fn_);
  }
  const user_op::DeviceAndStreamInferFn& device_and_stream_infer_fn() const {
    return device_and_stream_infer_fn_;
  }

  bool IsHostMemoryInput(int32_t input_index) const {
    return std::find(host_memory_input_ids_.begin(), host_memory_input_ids_.end(), input_index)
           != host_memory_input_ids_.end();
  }

  Maybe<void> InferPhysicalTensorDesc(
      const AttrMap& attrs, const std::string& device_tag,
      const std::function<const TensorMeta*(int32_t)>& TensorMeta4InputIndex,
      const std::function<TensorMeta*(int32_t)>& TensorMeta4OutputIndex) const;

  Maybe<void> InferLogicalTensorDesc(
      const AttrMap& attrs, Symbol<ParallelDesc> parallel_desc,
      const std::function<const TensorMeta*(int32_t)>& TensorMeta4InputIndex,
      const std::function<TensorMeta*(int32_t)>& TensorMeta4OutputIndex) const;
  Maybe<Symbol<Stream>> InferDeviceAndStream(const AttrMap& attrs, const TensorTuple& inputs,
                                             TensorTuple* outputs) const;
  LocalTensorInferCache* mut_local_tensor_infer_cache() const {
    return local_tensor_infer_cache_.get();
  }
  GlobalTensorInferCache* mut_global_tensor_infer_cache() const {
    return global_tensor_infer_cache_.get();
  }

 private:
  UserOpExpr(const std::string& op_name, UserOpConf&& proto, const AttrMap& base_attrs,
             const std::vector<std::string>& indexed_ibns,
             const std::vector<std::string>& indexed_obns);
  Maybe<void> Init(const std::shared_ptr<const UserOpExpr>& self);
  AttrMap base_attrs_;
  user_op::TensorDescInferFn logical_tensor_desc_infer_fn_;
  user_op::TensorDescInferFn physical_tensor_desc_infer_fn_;
  user_op::DataTypeInferFn dtype_infer_fn_;
  user_op::DeviceAndStreamInferFn device_and_stream_infer_fn_;
  mutable HashMap<Symbol<Stream>, std::shared_ptr<StatefulOpKernel>> stream2kernel_;
  std::shared_ptr<LocalTensorInferCache> local_tensor_infer_cache_;
  std::shared_ptr<GlobalTensorInferCache> global_tensor_infer_cache_;
  small_vector<int32_t> host_memory_input_ids_;
};

class GlobalToGlobalOpExpr : public OpExpr {
 public:
  virtual ~GlobalToGlobalOpExpr() = default;

  static Maybe<GlobalToGlobalOpExpr> New(const Optional<Symbol<NdSbp>>& grad_nd_sbp);

  const Optional<Symbol<NdSbp>>& grad_nd_sbp() const { return grad_nd_sbp_; }
  const std::string& op_type_name() const override;
  int input_size() const override { return 1; }
  int output_size() const override { return 1; }

  Maybe<bool> IsGradDisabled() const override { return false; }
  Maybe<bool> SupportNonContiguous() const override { return false; }
  Maybe<OpExprGradClosure> GetOrCreateOpGradClosure() const override;

 protected:
  GlobalToGlobalOpExpr(const Optional<Symbol<NdSbp>>& grad_nd_sbp);

  Optional<Symbol<NdSbp>> grad_nd_sbp_;  //  Reserved for configuring grad sbp
  mutable std::shared_ptr<OpExprGradFunctionIf> op_grad_func_;
};

class CastGlobalOpExpr : public OpExpr {
 public:
  virtual ~CastGlobalOpExpr() = default;

  const std::string& op_name() const { return op_name_; }
  int input_size() const override { return 1; }
  int output_size() const override { return 1; }

  Maybe<bool> IsGradDisabled() const override { return false; }
  Maybe<bool> SupportNonContiguous() const override { return false; }

 protected:
  CastGlobalOpExpr(const std::string& op_name);

  std::string op_name_;
  mutable std::shared_ptr<OpExprGradFunctionIf> op_grad_func_;
};

class LocalToGlobalOpExpr final : public CastGlobalOpExpr {
 public:
  ~LocalToGlobalOpExpr() = default;

  static Maybe<LocalToGlobalOpExpr> New(const std::string& op_name);

  const std::string& op_type_name() const override;
  Maybe<OpExprGradClosure> GetOrCreateOpGradClosure() const override;

 private:
  LocalToGlobalOpExpr(const std::string& op_name);
};

class GlobalToLocalOpExpr final : public CastGlobalOpExpr {
 public:
  ~GlobalToLocalOpExpr() = default;

  static Maybe<GlobalToLocalOpExpr> New(const std::string& op_name);

  const std::string& op_type_name() const override;
  Maybe<OpExprGradClosure> GetOrCreateOpGradClosure() const override;

 private:
  GlobalToLocalOpExpr(const std::string& op_name);
};

// NOTE(chengcheng): For Lazy nn.Graph Feed/Fetch EagerTensor to/from LazyTensor.
using FeedInputOpExpr = BuiltinOpExprImpl<FeedInputOpConf>;
using FeedVariableOpExpr = BuiltinOpExprImpl<FeedVariableOpConf>;
using FetchOutputOpExpr = BuiltinOpExprImpl<FetchOutputOpConf>;

// NOTE(chengcheng): Special SystemOp for image gpu decode.
using ImageDecoderRandomCropResizeOpExpr = BuiltinOpExprImpl<ImageDecoderRandomCropResizeOpConf>;

using VariableOpExpr = BuiltinOpExprImpl<VariableOpConf>;
using CastToLocalOpExpr = BuiltinOpExprImpl<CastToLocalOpConf>;
using CastFromLocalOpExpr = BuiltinOpExprImpl<CastFromLocalOpConf>;
using DistributeSplitOpExpr = BuiltinOpExprImpl<DistributeSplitOpConf>;
using DistributeCloneOpExpr = BuiltinOpExprImpl<DistributeCloneOpConf>;
using DistributeConcatOpExpr = BuiltinOpExprImpl<DistributeConcatOpConf>;
using DistributeAddOpExpr = BuiltinOpExprImpl<DistributeAddOpConf>;

class SelectTopNOpExpr final : public OpExpr {
 public:
  static Maybe<SelectTopNOpExpr> New() {
    return std::shared_ptr<SelectTopNOpExpr>(new SelectTopNOpExpr());
  }

  const std::string& op_type_name() const override {
    static const std::string kOpTypeName = "select_top_n";
    return kOpTypeName;
  }

  int input_size() const override {
    UNIMPLEMENTED();
    return 0;
  }

  int output_size() const override {
    // output should be resized in apply function
    return 0;
  }

  Maybe<bool> IsGradDisabled() const override { return false; }

  Maybe<bool> SupportNonContiguous() const override { return false; }

  Maybe<OpExprGradClosure> GetOrCreateOpGradClosure() const override;

 private:
  SelectTopNOpExpr() = default;

  mutable std::shared_ptr<OpExprGradFunctionIf> op_grad_func_;
};

class AutoGradCaptureState;

class FunctionOpExpr final : public OpExpr {
 public:
  using FType = AutogradFunctionBase::FType;
  FunctionOpExpr() = delete;
  static Maybe<FunctionOpExpr> New(const std::string& func_name, const FType& forward_fn,
                                   const FType& backward_fn) {
    return std::shared_ptr<FunctionOpExpr>(new FunctionOpExpr(func_name, forward_fn, backward_fn));
  }

  const std::string& op_type_name() const override { return func_name_; }

  int input_size() const override {
    PRINT_BUG_PROMPT_AND_ABORT() << "You cannot get input_size here.";
    return 0;
  }
  int output_size() const override {
    PRINT_BUG_PROMPT_AND_ABORT() << "You cannot get output_size here.";
    return 0;
  }

  FType forward() const { return forward_fn_; }
  FType backward() const { return backward_fn_; }

  std::shared_ptr<FunctionAutoGradCaptureState> state() const { return state_; }
  void reset_state() const;

  Maybe<bool> IsGradDisabled() const override { return false; }
  Maybe<bool> SupportNonContiguous() const override { return false; }
  Maybe<OpExprGradClosure> GetOrCreateOpGradClosure() const override;

 private:
  FunctionOpExpr(const std::string& func_name, const FType& forward_fn, const FType& backward_fn)
      : forward_fn_(forward_fn), backward_fn_(backward_fn), func_name_(func_name) {}

  FType forward_fn_;
  FType backward_fn_;
  std::string func_name_;
  mutable std::shared_ptr<FunctionAutoGradCaptureState> state_;
  mutable std::shared_ptr<OpExprGradFunctionIf> op_grad_func_;
};

}  // namespace one
}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_OP_EXPR_H_
