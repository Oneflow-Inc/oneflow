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
#ifndef ONEFLOW_USER_KERNELS_STATEFUL_LOCAL_OPKERNEL_H_
#define ONEFLOW_USER_KERNELS_STATEFUL_LOCAL_OPKERNEL_H_

#include "oneflow/core/eager/eager_blob_object.h"
#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/framework/op_kernel.h"
#include "oneflow/core/framework/user_op_kernel_registry.h"

namespace oneflow {

class AttrValueMap;

namespace eager {
struct LocalCallOpKernelUtil;
}  // namespace eager

namespace one {

class LocalUserKernelBaseContext;
class LocalUserKernelRegContext;
class LocalUserKernelCreateContext;
class LocalUserKernelInitContext;
class LocalUserOpInferContext;

using ArgVec = std::vector<std::pair<std::string, int32_t>>;

using EagerBlobObjectList =
    std::shared_ptr<const std::vector<std::shared_ptr<eager::EagerBlobObject>>>;

class EagerBlobObjectTensorView final : public user_op::Tensor {
 public:
  EagerBlobObjectTensorView(const std::function<eager::EagerBlobObject*()>& mut_eager_blob_object)
      : mut_eager_blob_object_(mut_eager_blob_object) {}

  const ShapeView& shape() const override { return mut_eager_blob_object_()->blob().shape(); }

  MutShapeView* mut_shape() override {
    return mut_eager_blob_object_()->mut_blob()->mut_shape_view();
  }

  DataType data_type() const override { return mut_eager_blob_object_()->blob().data_type(); }

  const MemoryCase& mem_case() const override {
    return mut_eager_blob_object_()->blob().mem_case();
  }

  const void* raw_dptr() const override { return mut_eager_blob_object_()->blob().dptr(); }

  void* mut_raw_dptr() override { return mut_eager_blob_object_()->mut_blob()->mut_dptr(); }

 private:
  const std::function<eager::EagerBlobObject*()> mut_eager_blob_object_;
};

class EagerBlobObjectTensorDescView final : public user_op::TensorDesc {
 public:
  EagerBlobObjectTensorDescView(
      const std::function<eager::EagerBlobObject*()>& mut_eager_blob_object)
      : mut_eager_blob_object_(mut_eager_blob_object) {}

  const Shape& shape() const override { return mut_eager_blob_object_()->blob_desc().shape(); }

  Shape* mut_shape() override { return &mut_eager_blob_object_()->mut_blob_desc()->mut_shape(); }

  DataType data_type() const override { return mut_eager_blob_object_()->blob_desc().data_type(); }

  DataType* mut_data_type() override {
    return mut_eager_blob_object_()->mut_blob_desc()->mut_data_type();
  }

  bool is_dynamic() const override { return mut_eager_blob_object_()->blob_desc().is_dynamic(); }

  bool* mut_is_dynamic() override {
    return mut_eager_blob_object_()->mut_blob_desc()->mut_is_dynamic();
  }

  void set_is_dynamic(bool val) override {
    mut_eager_blob_object_()->mut_blob_desc()->set_is_dynamic(val);
  }

 private:
  const std::function<eager::EagerBlobObject*()> mut_eager_blob_object_;
};

class ZeroCopyBaseContext {
 public:
  ZeroCopyBaseContext(const ArgVec* indexed_input_pairs, const ArgVec* indexed_output_pairs);

  user_op::TensorDesc* TensorDesc4ArgNameAndIndex(const std::string& arg_name, int32_t index) const;

  user_op::Tensor* Tensor4ArgNameAndIndex(const std::string& arg_name, int32_t index) const;

  const ArgVec& inputs() const { return *indexed_input_pairs_; }
  const ArgVec& outputs() const { return *indexed_output_pairs_; }

  void Update(EagerBlobObjectList inputs, EagerBlobObjectList outputs);

 private:
  const ArgVec* indexed_input_pairs_;
  const ArgVec* indexed_output_pairs_;
  std::map<std::string, std::vector<int32_t>> arg_name2bn_index2input_tensor_tuple_index_;
  std::map<std::string, std::vector<int32_t>> arg_name2bn_index2output_tensor_tuple_index_;
  std::vector<std::unique_ptr<EagerBlobObjectTensorView>> input_tensor_views_;
  std::vector<std::unique_ptr<EagerBlobObjectTensorView>> output_tensor_views_;
  std::vector<std::unique_ptr<EagerBlobObjectTensorDescView>> input_tensor_desc_views_;
  std::vector<std::unique_ptr<EagerBlobObjectTensorDescView>> output_tensor_desc_views_;
  EagerBlobObjectList input_tensors_;
  EagerBlobObjectList output_tensors_;
};

class LocalUserKernelBaseContext : public ZeroCopyBaseContext {
 public:
  LocalUserKernelBaseContext(const std::string& device_tag, const ArgVec* indexed_input_pairs,
                             const ArgVec* indexed_output_pairs);
  ~LocalUserKernelBaseContext() = default;

  DeviceType device_type() const { return device_type_; }
  const std::string& device_tag() const { return device_tag_; }
  const JobDesc& job_desc() const { UNIMPLEMENTED(); }

 private:
  const std::string device_tag_;
  const DeviceType device_type_;
};

class LocalUserOpInferContext : public user_op::InferContext {
 public:
  LocalUserOpInferContext(const user_op::UserOpConfWrapper* user_op_conf,
                          const ArgVec* index_input_pairs, const ArgVec* indexed_output_pairs);
  ~LocalUserOpInferContext() override = default;

  const user_op::TensorDesc* LogicalTensorDesc4ArgNameAndIndex(const std::string& arg_name,
                                                               int32_t index) const override {
    UNIMPLEMENTED();
  }
  user_op::TensorDesc* TensorDesc4ArgNameAndIndex(const std::string& arg_name,
                                                  int32_t index) override;
  Shape* Shape4ArgNameAndIndex(const std::string& arg_name, int32_t index) override {
    return TensorDesc4ArgNameAndIndex(arg_name, index)->mut_shape();
  }
  DataType* Dtype4ArgNameAndIndex(const std::string& arg_name, int32_t index) override {
    return TensorDesc4ArgNameAndIndex(arg_name, index)->mut_data_type();
  }
  bool* IsDynamic4ArgNameAndIndex(const std::string& arg_name, int32_t index) override {
    return TensorDesc4ArgNameAndIndex(arg_name, index)->mut_is_dynamic();
  }

  const ArgVec& inputs() const override { return zero_copy_base_ctx_.inputs(); }
  const ArgVec& outputs() const override { return zero_copy_base_ctx_.outputs(); }
  const JobDesc* job_desc() const override {
    UNIMPLEMENTED();
    return nullptr;
  }
  const ParallelContext& parallel_ctx() const override { UNIMPLEMENTED(); };
  const ParallelDesc& parallel_desc() const override { UNIMPLEMENTED(); }
  const SbpParallel& SbpParallel4ArgNameAndIndex(const std::string& arg_name,
                                                 int32_t index) const override {
    UNIMPLEMENTED();
  }
  const ParallelDistribution& ParallelDistribution4ArgNameAndIndex(const std::string& arg_name,
                                                                   int32_t index) const override {
    UNIMPLEMENTED();
  }

  int64_t parallel_num() const override { return 1; }

  void Update(EagerBlobObjectList inputs, EagerBlobObjectList outputs);

 private:
  const user_op::UserOpConfWrapper& user_op_conf() const override { return *user_op_conf_; }
  const std::shared_ptr<user_op::AttrVal>& Attr4AttrName(
      const std::string& attr_name) const override {
    return user_op_conf().Attr4AttrName(attr_name);
  }

  const user_op::UserOpConfWrapper* user_op_conf_;
  ZeroCopyBaseContext zero_copy_base_ctx_;
};

class LocalUserKernelComputeContext final : public user_op::KernelComputeContext {
 public:
  explicit LocalUserKernelComputeContext(DeviceCtx* device_ctx, const std::string& device_tag,
                                         const user_op::UserOpConfWrapper* user_op_conf,
                                         const ArgVec* index_input_pairs,
                                         const ArgVec* indexed_output_pairs);
  ~LocalUserKernelComputeContext() = default;

  const user_op::TensorDesc* TensorDesc4ArgNameAndIndex(const std::string& arg_name,
                                                        int32_t index) const override {
    return base_ctx_.TensorDesc4ArgNameAndIndex(arg_name, index);
  }

  user_op::Tensor* Tensor4ArgNameAndIndex(const std::string& arg_name, int32_t index) override {
    return base_ctx_.Tensor4ArgNameAndIndex(arg_name, index);
  }
  DeviceCtx* device_ctx() override { return device_ctx_; }

  DeviceType device_type() const override { return base_ctx_.device_type(); }
  const ParallelContext& parallel_ctx() const override { UNIMPLEMENTED(); };
  const JobDesc& job_desc() const override { UNIMPLEMENTED(); };

  const ArgVec& inputs() const override { return base_ctx_.inputs(); };
  const ArgVec& outputs() const override { return base_ctx_.outputs(); };

  void Update(EagerBlobObjectList inputs, EagerBlobObjectList outputs, DeviceCtx* device_ctx);

 private:
  const user_op::UserOpConfWrapper& user_op_conf() const override { return *user_op_conf_; }
  const std::shared_ptr<user_op::AttrVal>& Attr4AttrName(
      const std::string& attr_name) const override {
    return user_op_conf().Attr4AttrName(attr_name);
  }

  const user_op::UserOpConfWrapper* user_op_conf_;
  DeviceCtx* device_ctx_;
  LocalUserKernelBaseContext base_ctx_;
};

class StatefulOpKernel final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(StatefulOpKernel);
  static Maybe<StatefulOpKernel> New(const std::shared_ptr<OperatorConf>& op_conf,
                                     const std::shared_ptr<MemoryCase>& mem_case,
                                     const std::shared_ptr<const ParallelDesc>& parallel_desc,
                                     const std::shared_ptr<ArgVec> indexed_input_pairs,
                                     const std::shared_ptr<ArgVec> indexed_output_pairs);
  ~StatefulOpKernel();
  const std::shared_ptr<MemoryCase> mem_case() const { return mem_case_; };
  const std::vector<int64_t>& input_tuple_indexes4const_ibns() const {
    return input_tuple_indexes4const_ibns_;
  }
  const std::vector<int64_t>& input_tuple_indexes4mut_ibns() const {
    return input_tuple_indexes4mut_ibns_;
  }
  const std::vector<int64_t>& output_tuple_indexes4mut_obns() const {
    return output_tuple_indexes4mut_obns_;
  }
  const std::vector<int64_t>& output_tuple_indexes4mut2_obns() const {
    return output_tuple_indexes4mut2_obns_;
  }

  std::shared_ptr<VmLocalDepObject> infer_local_dep_object() const {
    return infer_local_dep_object_;
  }
  std::shared_ptr<VmLocalDepObject> compute_local_dep_object() const {
    return compute_local_dep_object_;
  }

  void InferDataType(EagerBlobObjectList inputs, EagerBlobObjectList outputs) {
    data_type_infer_fn_(UpdateInferContext(inputs, outputs));
    UpdateInferContext(nullptr, nullptr);
  }

  void UpdateOpAttrs(const AttrValueMap& attrs);

 private:
  friend struct eager::LocalCallOpKernelUtil;
  StatefulOpKernel() = default;
  LocalUserOpInferContext* UpdateInferContext(EagerBlobObjectList inputs,
                                              EagerBlobObjectList outputs);
  LocalUserKernelComputeContext* UpdateComputeContext(EagerBlobObjectList inputs,
                                                      EagerBlobObjectList outputs,
                                                      DeviceCtx* device_ctx);

  user_op::TensorDescInferFn TensorDescInferFn() const;
  user_op::DataTypeInferFn DataTypeInferFn() const;

  void TryInitOpKernelState(const user_op::OpKernel* op_kernel, DeviceCtx* device_ctx,
                            EagerBlobObjectList inputs, EagerBlobObjectList outputs,
                            user_op::OpKernelState** state);

  eager::EagerBlobObject* mut_temp_blob_object();

  user_op::OpKernelState* mut_opkernel_state(const user_op::OpKernel* opkernel) {
    return op_kernel_state_map_.at(opkernel).get();
  }

  bool need_check_mem_case() const { return need_check_mem_case_; }
  void set_need_check_mem_case(bool value) { need_check_mem_case_ = value; }

  Maybe<const user_op::OpKernel*> ChooseOpKernel(EagerBlobObjectList inputs,
                                                 EagerBlobObjectList outputs);

  const user_op::InferTmpSizeFn& GetInferTmpSizeFn(const user_op::OpKernel* op_kernel) const;

  std::shared_ptr<OperatorConf> op_conf_;
  std::unique_ptr<user_op::UserOpConfWrapper> user_op_conf_;
  std::shared_ptr<MemoryCase> mem_case_;
  std::unique_ptr<LocalUserKernelRegContext> reg_ctx_;
  std::unique_ptr<LocalUserKernelCreateContext> create_ctx_;
  std::unique_ptr<LocalUserOpInferContext> op_infer_ctx_;
  std::unique_ptr<LocalUserKernelComputeContext> compute_ctx_;
  std::shared_ptr<ArgVec> indexed_input_pairs_;
  std::shared_ptr<ArgVec> indexed_output_pairs_;
  bool need_check_mem_case_;
  user_op::TensorDescInferFn tensor_desc_infer_fn_;
  user_op::DataTypeInferFn data_type_infer_fn_;
  HashMap<const user_op::OpKernelRegistryResult*, std::shared_ptr<const user_op::OpKernel>>
      op_kernel_map_;
  HashMap<const user_op::OpKernel*, std::shared_ptr<user_op::OpKernelState>> op_kernel_state_map_;
  HashMap<const user_op::OpKernel*, const user_op::InferTmpSizeFn*> infer_tmp_size_fn_map_;
  std::unique_ptr<eager::EagerBlobObject> tmp_blob_object_;
  std::vector<int64_t> input_tuple_indexes4const_ibns_;
  std::vector<int64_t> input_tuple_indexes4mut_ibns_;
  std::vector<int64_t> output_tuple_indexes4mut_obns_;
  std::vector<int64_t> output_tuple_indexes4mut2_obns_;
  std::shared_ptr<VmLocalDepObject> infer_local_dep_object_;
  std::shared_ptr<VmLocalDepObject> compute_local_dep_object_;
};

}  // namespace one

}  // namespace oneflow

#endif  // ONEFLOW_USER_KERNELS_STATEFUL_LOCAL_OPKERNEL_H_
