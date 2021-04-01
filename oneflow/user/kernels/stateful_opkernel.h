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
#ifndef ONEFLOW_CORE_KERNEL_EAGER_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_EAGER_KERNEL_H_

#include "oneflow/core/eager/eager_blob_object.h"
#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/framework/op_kernel.h"

namespace oneflow {

namespace eager {
class LocalCallOpKernelUtil;
class EagerBlobObject;
}  // namespace eager

namespace one {

class LocalUserKernelInitContext;
class LocalUserOpInferContext;
class LocalUserKernelComputeContext;

using TensorsPtr = std::vector<std::shared_ptr<eager::EagerBlobObject>>*;
using TensorIndexMap = std::shared_ptr<HashMap<std::string, std::vector<int64_t>>>;

class StatefulOpKernel final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(StatefulOpKernel);
  StatefulOpKernel(const std::shared_ptr<const JobDesc> job_desc, const KernelConf& kernel_conf,
                   TensorIndexMap bn_in_op2bn_index2input_tensor_index,
                   TensorIndexMap bn_in_op2bn_index2output_tensor_index);
  ~StatefulOpKernel() = default;

  LocalUserOpInferContext* UpdateInferContext(TensorsPtr inputs, TensorsPtr outputs);
  LocalUserKernelComputeContext* UpdateComputeContext(TensorsPtr inputs, TensorsPtr outputs,
                                                      DeviceCtx* device_ctx);

  user_op::TensorDescInferFn TensorDescInferFn() const;

  Maybe<void> TryInitOpKernelState(DeviceCtx* device_ctx);

  eager::EagerBlobObject mut_temp_blob_object() { TODO(); };

  std::shared_ptr<user_op::OpKernelState> mut_opkernel_state() { return state_; }

  const MemoryCase& mem_case() { TODO(); };
  bool need_check_mem_case() const { return need_check_mem_case_; }
  void set_need_check_mem_case(bool value) { need_check_mem_case_ = value; }

  const std::shared_ptr<const JobDesc> job_desc() const { return job_desc_; };
  const KernelConf& kernel_conf() const { return kernel_conf_; }

 private:
  friend class eager::LocalCallOpKernelUtil;
  void InitOpKernel(const KernelConf& kernel_conf);
  std::shared_ptr<const JobDesc> job_desc_;
  const KernelConf& kernel_conf_;
  std::unique_ptr<const user_op::OpKernel> kernel_;
  std::unique_ptr<LocalUserOpInferContext> op_infer_ctx_;
  std::unique_ptr<LocalUserKernelComputeContext> compute_ctx_;
  std::unique_ptr<LocalUserKernelInitContext> init_ctx_;
  std::shared_ptr<user_op::OpKernelState> state_;
  TensorIndexMap bn_in_op2bn_index2input_tensor_index_;
  TensorIndexMap bn_in_op2bn_index2output_tensor_index_;
  bool need_check_mem_case_;
  user_op::TensorDescInferFn tensor_desc_infer_fn_;
};

}  // namespace one

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_EAGER_KERNEL_H_
