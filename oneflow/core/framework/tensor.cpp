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
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/tensor_rpc_util.h"
#include "oneflow/core/framework/nd_sbp.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/job/job_build_and_infer_ctx_mgr.h"
#include "oneflow/core/job/job_build_and_infer_ctx.h"
#include "oneflow/core/framework/device.h"
#include "oneflow/core/framework/dtype.h"
#include "oneflow/core/framework/instructions_builder.h"
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/core/autograd/autograd_engine.h"
#include "oneflow/core/framework/op_interpreter/eager_mirrored_op_interpreter.h"
#include "oneflow/core/framework/tensor_rpc_util.h"
#include "oneflow/core/functional/functional.h"

namespace oneflow {

namespace one {

Parameter::Parameter(std::shared_ptr<Tensor> tensor, bool requires_grad) {
  while (auto parameter = std::dynamic_pointer_cast<Parameter>(tensor)) {
    tensor = parameter->tensor_;
  }
  this->tensor_ = tensor->detach().GetPtrOrThrow();
  // this->tensor_ = std::move(tensor);
  // TODO: in `y = flow.nn.Parameter(x)`, y should have its own "requires_grad" field
  // (align with PyTorch) instead of sharing it with x
  CHECK_JUST(this->tensor_->set_requires_grad(requires_grad));
  auto blob_object = CHECK_JUST(tensor_->eager_blob_object());
  if (auto dtr_eager_blob_object = std::dynamic_pointer_cast<vm::DTREagerBlobObject>(blob_object)) {
    dtr_eager_blob_object->set_evict_attr(false);
  }
}

Maybe<MirroredTensor> Parameter::AsMirroredTensor() {
  if (const auto& mirrored_tensor = std::dynamic_pointer_cast<MirroredTensor>(tensor_)) {
    return mirrored_tensor;
  }
  RETURN_ERROR_WITH_BUG_PROMPT();
}

Maybe<void> Parameter::set_data(const std::shared_ptr<Tensor>& other) {
  CHECK_OR_RETURN(is_local() == other->is_local() && is_eager() == other->is_eager())
      << "You can't assign copy between tensors with different type";
  bool old_requires_grad = tensor_->requires_grad();
  this->tensor_ = JUST(other->detach());
  JUST(this->tensor_->set_requires_grad(old_requires_grad));

  auto blob_object = JUST(this->tensor_->eager_blob_object());
  if (auto dtr_eager_blob_object = std::dynamic_pointer_cast<vm::DTREagerBlobObject>(blob_object)) {
    dtr_eager_blob_object->set_evict_attr(false);
  }
  return Maybe<void>::Ok();
}

Maybe<void> DTRMirroredTensor::set_tensor_inputs(const TensorTuple& inputs) {
  if (oneflow::DTRDebugEnabled()) {
    std::stringstream ss;
    ss << "set inputs of " << this << " (ebo " << JUST(eager_blob_object()).get() << ") to ";
    for (const auto& x : inputs) {
      ss << x.get() << " (ebo " << JUST(x->eager_blob_object()).get() << "), ";
    }
    LOG(INFO) << ss.str();
  }
  std::vector<std::shared_ptr<Holder>> input_holders;
  for (const auto& x : inputs) {
    if (auto dtr_mirrored_tensor =
            std::dynamic_pointer_cast<DTRMirroredTensor>(JUST(x->AsMirroredTensor()))) {
      const auto& input_holder = dtr_mirrored_tensor->holder();
      CHECK_NOTNULL_OR_RETURN(input_holder);
      input_holders.push_back(input_holder);
    } else {
      LOG(INFO) << "no dtr ebo, ebo " << JUST(x->eager_blob_object()).get() << ", real type "
                << typeid(*x).name() << ", " << typeid(*JUST(x->AsMirroredTensor())).name()
                << ", bug?" << std::endl;
      // do nothing
    }
  }
  holder_ =
      std::make_shared<Holder>(input_holders, JUST(tensor_storage()), JUST(eager_blob_object()));
  if (oneflow::DTRDebugEnabled()) { LOG(INFO) << "set_tenosr_inputs done"; }
  return Maybe<void>::Ok();
}

Maybe<MirroredTensor> StaticZerosTensor::AsMirroredTensor() {
  CHECK_OR_RETURN(is_local());
  return std::dynamic_pointer_cast<MirroredTensor>(
      JUST(functional::Constant(*shape_, Scalar(0), JUST(DType::Get(dtype_)), device_)));
}

/* static */ Maybe<MirroredTensor> MirroredTensor::MakeTensor(
    const std::shared_ptr<const Shape>& shape, DataType dtype, const Symbol<Device>& device,
    bool is_lazy, bool requires_grad, bool is_leaf) {
  const auto& tensor_meta =
      std::make_shared<MirroredTensorMeta>(std::make_shared<Shape>(*shape), dtype, device);
  if (is_lazy) {
    const auto& impl =
        std::make_shared<LazyMirroredTensorImpl>(tensor_meta, requires_grad, is_leaf);
    return std::make_shared<MirroredTensor>(impl);
  } else if (oneflow::DTREnabled()) {
    const auto& impl =
        std::make_shared<DTREagerMirroredTensorImpl>(tensor_meta, requires_grad, is_leaf);
    const auto& tensor = std::make_shared<DTRMirroredTensor>(impl);
    const auto& outputs = std::make_shared<TensorTuple>();
    outputs->push_back(tensor);
    JUST(RunEmptyOp(outputs.get()));
    return static_cast<std::shared_ptr<MirroredTensor>>(tensor);
  } else {
    const auto& impl =
        std::make_shared<EagerMirroredTensorImpl>(tensor_meta, requires_grad, is_leaf);
    return std::make_shared<MirroredTensor>(impl);
  }
}

bool MirroredTensor::is_cuda() const { return CHECK_JUST(device())->type() == "cuda"; }

Maybe<Tensor> MirroredTensor::detach() const {
  std::shared_ptr<Tensor> tensor = std::make_shared<MirroredTensor>(JUST(impl_->detach()));
  return tensor;
}

Maybe<Tensor> DTRMirroredTensor::detach() const {
  auto tensor = std::make_shared<DTRMirroredTensor>(
      CHECK_NOTNULL(std::dynamic_pointer_cast<DTREagerMirroredTensorImpl>(JUST(impl_->detach()))));
  tensor->holder_ = this->holder_;
  return std::dynamic_pointer_cast<Tensor>(tensor);
}

Maybe<Tensor> MirroredTensor::clone() const {
  const auto& device_type = JUST(this->device())->type();
  int64_t device_id = JUST(this->device())->device_id();
  std::shared_ptr<MirroredTensor> input =
      std::const_pointer_cast<MirroredTensor>(shared_from_this());
  return JUST(functional::Copy(input, device_type, device_id));
}

Maybe<void> DTRMirroredTensor::set_blob_object_bp_required() {
  auto blob_object = JUST(eager_blob_object());
  if (auto dtr_eager_blob_object = std::dynamic_pointer_cast<vm::DTREagerBlobObject>(blob_object)) {
    // if (auto* dtr_eager_blob_object = dynamic_cast<vm::DTREagerBlobObject*>(blob_object.get())) {
    dtr_eager_blob_object->set_bp_required(true);
  }
  return Maybe<void>::Ok();
}

Maybe<Tensor> ConsistentTensor::clone() const {
  const auto& local_tensor = JUST(cur_rank_phy_tensor());
  const auto& device_type = JUST(local_tensor->device())->type();
  int64_t device_id = JUST(local_tensor->device())->device_id();
  const auto& cloned_local_tensor = JUST(functional::Copy(local_tensor, device_type, device_id));
  DisableCheckConsistentTensorMetaScope disable_meta_check{};
  return functional::LocalToConsistent(cloned_local_tensor, JUST(parallel_desc()),
                                       *JUST(GetSbpList(JUST(nd_sbp()))), *shape(), dtype());
}

Maybe<ConsistentTensor> ConsistentTensor::MakeTensor(const std::shared_ptr<const Shape>& shape,
                                                     DataType dtype, Symbol<cfg::NdSbp> nd_sbp,
                                                     Symbol<ParallelDesc> parallel_desc,
                                                     bool is_lazy, bool requires_grad,
                                                     bool is_leaf) {
  std::shared_ptr<ConsistentTensorImpl> impl;
  Symbol<ConsistentTensorMeta> consistent_tensor_meta(
      ConsistentTensorMeta(shape, dtype, nd_sbp, parallel_desc));
  if (is_lazy) {
    impl =
        std::make_shared<LazyConsistentTensorImpl>(consistent_tensor_meta, requires_grad, is_leaf);
  } else {
    impl = JUST(EagerConsistentTensorImpl::New(consistent_tensor_meta, requires_grad, is_leaf));
  }
  return std::make_shared<ConsistentTensor>(impl);
}

bool ConsistentTensor::is_cuda() const {
  return CHECK_JUST(parallel_desc())->device_type() == DeviceType::kGPU;
}

Maybe<Tensor> ConsistentTensor::detach() const {
  std::shared_ptr<Tensor> t = std::make_shared<ConsistentTensor>(impl_);
  return t;
}

}  // namespace one

}  // namespace oneflow
