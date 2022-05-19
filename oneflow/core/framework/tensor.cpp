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
#include "oneflow/core/framework/tensor_methods.h"
#include "oneflow/core/framework/tensor_rpc_util.h"
#include "oneflow/core/framework/nd_sbp.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/eager/dtr_eager_blob_object.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/job/job_build_and_infer_ctx_mgr.h"
#include "oneflow/core/job/job_build_and_infer_ctx.h"
#include "oneflow/core/framework/device.h"
#include "oneflow/core/framework/dtype.h"
#include "oneflow/core/framework/instructions_builder.h"
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/core/autograd/autograd_engine.h"
#include "oneflow/core/framework/op_interpreter/eager_mirrored_op_interpreter.h"
#include "oneflow/core/functional/functional.h"

namespace oneflow {

namespace one {

Parameter::Parameter(const std::shared_ptr<Tensor> &tensor, bool requires_grad) : ProxyTensor<Parameter>(tensor) {
  // TODO: in `y = flow.nn.Parameter(x)`, y should have its own "requires_grad" field
  // (align with PyTorch) instead of sharing it with x
  CHECK_JUST(this->tensor_->set_requires_grad(requires_grad));
  auto blob_object = CHECK_JUST(tensor_->eager_blob_object());
  if (auto dtr_eager_blob_object = std::dynamic_pointer_cast<vm::DTREagerBlobObject>(blob_object)) {
    dtr_eager_blob_object->set_evict_attr(false);
  }
}

Maybe<void> Parameter::set_data(const std::shared_ptr<Tensor>& other) {
  JUST(ProxyTensor<Parameter>::set_data(other));

  auto blob_object = JUST(this->tensor_->eager_blob_object());
  if (auto dtr_eager_blob_object = std::dynamic_pointer_cast<vm::DTREagerBlobObject>(blob_object)) {
    dtr_eager_blob_object->set_evict_attr(false);
  }
  return Maybe<void>::Ok();
}

bool DTRMirroredTensor::is_in_memory() const {
  return std::dynamic_pointer_cast<vm::DTREagerBlobObject>(CHECK_JUST(eager_blob_object()))
      ->is_in_memory();
  ;
}

Maybe<void> DTRMirroredTensor::set_tensor_inputs(const TensorTuple& inputs) {
  if (dtr::debug_level() >= 2) {
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
  if (dtr::debug_level() >= 3) { LOG(INFO) << "set_tenosr_inputs done"; }
  return Maybe<void>::Ok();
}

Maybe<MirroredTensor> StaticZerosTensor::AsMirroredTensor() {
  CHECK_OR_RETURN(is_local());
  return std::dynamic_pointer_cast<MirroredTensor>(
      JUST(functional::Constant(*shape_, Scalar(0), JUST(DType::Get(dtype_)), device_)));
}

std::shared_ptr<Tensor> Parameter::contiguous() const {
  const auto& tensor = std::const_pointer_cast<Tensor>(shared_from_this());
  if (tensor_->is_contiguous()) { return tensor; }
  return CHECK_JUST(functional::ToContiguous(tensor));
}

std::shared_ptr<Tensor> Parameter::pin_memory() const {
  std::shared_ptr<Tensor> tensor = std::const_pointer_cast<Tensor>(shared_from_this());
  return CHECK_JUST(functional::PinMemory(tensor));
}

/* static */ Maybe<MirroredTensor> MirroredTensor::MakeTensor(
    const std::shared_ptr<const Shape>& shape, const std::shared_ptr<const Stride>& stride,
    DataType dtype, const Symbol<Device>& device, bool is_lazy, bool requires_grad, bool is_leaf) {
  const auto& tensor_meta =
      std::make_shared<MirroredTensorMeta>(std::make_shared<Shape>(*shape), dtype, device);
  if (is_lazy) {
    const auto& impl =
        std::make_shared<LazyMirroredTensorImpl>(tensor_meta, requires_grad, is_leaf);
    return std::make_shared<MirroredTensor>(impl);
  } else if (dtr::is_enabled()) {
    const auto& impl =
        std::make_shared<DTREagerMirroredTensorImpl>(tensor_meta, requires_grad, is_leaf);
    const auto& tensor = std::make_shared<DTRMirroredTensor>(impl);
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

Maybe<void> DTRMirroredTensor::set_blob_object_bp_required() {
  auto blob_object = JUST(eager_blob_object());
  if (auto dtr_eager_blob_object = std::dynamic_pointer_cast<vm::DTREagerBlobObject>(blob_object)) {
    // if (auto* dtr_eager_blob_object = dynamic_cast<vm::DTREagerBlobObject*>(blob_object.get())) {
    dtr_eager_blob_object->set_bp_required(true);
  }
  return Maybe<void>::Ok();
}

std::shared_ptr<Tensor> MirroredTensor::contiguous() const {
  std::shared_ptr<Tensor> tensor = std::const_pointer_cast<Tensor>(shared_from_this());
  if (tensor->is_contiguous()) { return tensor; }
  return CHECK_JUST(functional::ToContiguous(tensor));
}

std::shared_ptr<Tensor> MirroredTensor::pin_memory() const {
  std::shared_ptr<Tensor> tensor = std::const_pointer_cast<Tensor>(shared_from_this());
  return CHECK_JUST(functional::PinMemory(tensor));
}

Maybe<Tensor> MirroredTensor::clone() const {
  const auto& device_type = JUST(this->device())->type();
  int64_t device_id = JUST(this->device())->device_id();
  std::shared_ptr<Tensor> input = std::const_pointer_cast<Tensor>(shared_from_this());
  const bool pin_memory = JUST(JUST(input->AsMirroredTensor())->eager_blob_object())->pin_memory();
  return JUST(functional::Copy(input, device_type, device_id, /*pin_memory=*/pin_memory));
}

std::shared_ptr<Tensor> ConsistentTensor::contiguous() const {
  std::shared_ptr<Tensor> tensor = std::const_pointer_cast<Tensor>(shared_from_this());
  if (tensor->is_contiguous()) { return tensor; }
  return CHECK_JUST(functional::ToContiguous(tensor));
}

std::shared_ptr<Tensor> ConsistentTensor::pin_memory() const {
  std::shared_ptr<Tensor> tensor = std::const_pointer_cast<Tensor>(shared_from_this());
  return CHECK_JUST(functional::PinMemory(tensor));
}

Maybe<Tensor> ConsistentTensor::clone() const {
  const auto& local_tensor = JUST(cur_rank_phy_tensor());
  const auto& device_type = JUST(local_tensor->device())->type();
  int64_t device_id = JUST(local_tensor->device())->device_id();
  const auto& cloned_local_tensor =
      JUST(functional::Copy(local_tensor, device_type, device_id, /*pin_memory=*/false));
  DisableCheckConsistentTensorMetaScope disable_meta_check{};
  return functional::LocalToConsistent(cloned_local_tensor, JUST(parallel_desc()),
                                       *JUST(GetSbpList(JUST(nd_sbp()))), *shape(), dtype());
}

Maybe<ConsistentTensor> ConsistentTensor::MakeTensor(const std::shared_ptr<const Shape>& shape,
                                                     DataType dtype, Symbol<NdSbp> nd_sbp,
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
  return CHECK_JUST(parallel_desc())->device_type() == DeviceType::kCUDA;
}

Maybe<Tensor> ConsistentTensor::detach() const {
  std::shared_ptr<Tensor> tensor = std::make_shared<ConsistentTensor>(JUST(impl_->detach()));
  return tensor;
}

Maybe<void> ConsistentTensor::set_data(const std::shared_ptr<Tensor>& other) {
  CHECK_OR_RETURN(this->is_leaf())
      << "Only leaf tensor's data can be set, because non-leaf tensor's data has been captured in "
         "the backward graph in autograd.";
  const auto& consistent_tensor =
      std::dynamic_pointer_cast<ConsistentTensor>(JUST(other->detach()));
  CHECK_NOTNULL_OR_RETURN(consistent_tensor);
  JUST(WithConsistencyChecked(consistent_tensor,
                              [&]() -> Maybe<void> { return Maybe<void>::Ok(); }));

  bool old_requires_grad = requires_grad();
  impl_ = consistent_tensor->impl_;
  JUST(set_requires_grad(old_requires_grad));
  grad_fn_node_ = nullptr;
  return Maybe<void>::Ok();
}

}  // namespace one

}  // namespace oneflow
