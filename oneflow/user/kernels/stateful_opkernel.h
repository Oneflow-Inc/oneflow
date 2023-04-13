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
#ifndef ONEFLOW_USER_KERNELS_STATEFUL_OPKERNEL_H_
#define ONEFLOW_USER_KERNELS_STATEFUL_OPKERNEL_H_

#include "oneflow/core/eager/eager_blob_object.h"
#include "oneflow/core/common/tensor_meta.h"
#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/framework/op_kernel.h"
#include "oneflow/core/framework/stream.h"
#include "oneflow/core/framework/user_op_kernel_registry.h"
#include "oneflow/core/framework/arg_tuple.h"
#include "oneflow/core/framework/op_interpreter.h"
#include "oneflow/core/common/op_args_vector.h"

namespace oneflow {

class AttrMap;

namespace vm {
struct OpCallInstructionUtil;
}

namespace eager {
class CallContext;
}

namespace one {

using ArgVec = std::vector<std::pair<std::string, int32_t>>;

class UserKernelRegContextHelper;
class UserOpInferContextHelper;
class UserKernelInitAndCacheContextHelper;
class UserKernelComputeContextHelper;

class StatefulOpKernel final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(StatefulOpKernel);
  static Maybe<StatefulOpKernel> New(const std::shared_ptr<OperatorConf>& op_conf,
                                     const Symbol<Stream>& stream, const AttrMap& base_attrs,
                                     const std::shared_ptr<const ParallelDesc>& parallel_desc,
                                     const std::shared_ptr<const ArgTuple>& input_arg_tuple,
                                     const std::shared_ptr<const ArgTuple>& output_arg_tuple);
  ~StatefulOpKernel();
  const Symbol<Stream>& stream() const { return stream_; }
  const std::shared_ptr<MemoryCase>& mem_case() const { return stream_->device()->mem_case(); }
  const std::string& op_type_name() const { return op_conf_->user_conf().op_type_name(); }
  const OpArgsVector<int64_t>& input_tuple_indexes4const_ibns() const {
    return input_tuple_indexes4const_ibns_;
  }
  const OpArgsVector<int64_t>& input_tuple_indexes4mut_ibns() const {
    return input_tuple_indexes4mut_ibns_;
  }
  const OpArgsVector<int64_t>& output_tuple_indexes4mut_obns() const {
    return output_tuple_indexes4mut_obns_;
  }
  const OpArgsVector<int64_t>& output_tuple_indexes4mut2_obns() const {
    return output_tuple_indexes4mut2_obns_;
  }

  bool output_is_mut2_type(int64_t index) const {
    return output_tuple_indexes2is_mut2_type_.at(index);
  }

  const AttrMap& base_attrs() const { return base_attrs_; }

  size_t InferTmpSize(eager::CallContext* call_ctx, const user_op::OpKernel* user_opkernel) const;

  Maybe<void> ChooseOpKernel(eager::CallContext* call_ctx, const user_op::OpKernel** user_opkernel,
                             bool* need_temp_storage);

  const OperatorConf& op_conf() const { return *op_conf_; }

 private:
  friend struct vm::OpCallInstructionUtil;
  StatefulOpKernel() = default;

  void Compute(eager::CallContext* call_ctx, ep::Stream* stream,
               const user_op::OpKernel* user_opkernel, user_op::OpKernelState* state,
               const user_op::OpKernelCache* cache) const;

  user_op::TensorDescInferFn TensorDescInferFn() const;
  user_op::DataTypeInferFn DataTypeInferFn() const;

  void TryInitOpKernelStateAndCache(eager::CallContext* call_ctx, ep::Stream* stream,
                                    const user_op::OpKernel* op_kernel,
                                    user_op::OpKernelState** state, user_op::OpKernelCache** cache);

  user_op::OpKernelState* mut_opkernel_state(const user_op::OpKernel* opkernel) {
    return op_kernel_state_map_.at(opkernel).get();
  }

  const user_op::InferTmpSizeFn& GetInferTmpSizeFn(const user_op::OpKernel* op_kernel) const;

  std::shared_ptr<OperatorConf> op_conf_;
  AttrMap base_attrs_;
  std::unique_ptr<user_op::UserOpConfWrapper> user_op_conf_;
  Symbol<Stream> stream_;
  std::unique_ptr<const UserKernelRegContextHelper> reg_ctx_helper_;
  std::unique_ptr<const UserOpInferContextHelper> op_infer_ctx_helper_;
  std::unique_ptr<const UserKernelInitAndCacheContextHelper> init_and_cache_ctx_helper_;
  std::unique_ptr<const UserKernelComputeContextHelper> compute_ctx_helper_;
  std::shared_ptr<const ArgTuple> input_arg_tuple_;
  std::shared_ptr<const ArgTuple> output_arg_tuple_;
  user_op::TensorDescInferFn tensor_desc_infer_fn_;
  user_op::DataTypeInferFn data_type_infer_fn_;
  // NOTE: every device has its own stateful local opkernel instance,
  // so only group kernels by dtype
  std::array<std::vector<std::pair<const user_op::OpKernelRegistryResult*,
                                   std::shared_ptr<const user_op::OpKernel>>>,
             DataType_ARRAYSIZE>
      dtype2cached_kernels_;
  HashMap<const user_op::OpKernel*, std::shared_ptr<user_op::OpKernelState>> op_kernel_state_map_;
  HashMap<const user_op::OpKernel*, std::shared_ptr<user_op::OpKernelCache>> op_kernel_cache_map_;
  HashMap<const user_op::OpKernel*, const user_op::InferTmpSizeFn*> infer_tmp_size_fn_map_;
  OpArgsVector<int64_t> input_tuple_indexes4const_ibns_;
  OpArgsVector<int64_t> input_tuple_indexes4mut_ibns_;
  OpArgsVector<int64_t> output_tuple_indexes4mut_obns_;
  OpArgsVector<int64_t> output_tuple_indexes4mut2_obns_;
  OpArgsVector<bool> output_tuple_indexes2is_mut2_type_;
};

}  // namespace one

}  // namespace oneflow

#endif  // ONEFLOW_USER_KERNELS_STATEFUL_OPKERNEL_H_
