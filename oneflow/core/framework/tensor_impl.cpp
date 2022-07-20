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
#include <type_traits>
#include "oneflow/core/common/blocking_then_busy.h"
#include "oneflow/core/common/stream_role.h"
#include "oneflow/core/framework/tensor_meta.h"
#include "oneflow/core/vm/virtual_machine.h"
#include "oneflow/core/framework/instructions_builder.h"
#include "oneflow/core/framework/tensor_impl.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/common/stride.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/job/sbp_parallel.h"
#include "oneflow/core/functional/functional.h"
#include "oneflow/core/framework/dtype.h"
#include "oneflow/core/eager/eager_blob_object.h"
#include "oneflow/core/eager/local_dep_object.h"
#include "oneflow/core/vm/vm_util.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/control/global_process_ctx.h"
#include "oneflow/core/register/ofblob.h"
#include "oneflow/core/framework/stream_allocator_is_pinned.h"

namespace oneflow {
namespace one {

Maybe<void> TensorImpl::set_requires_grad(bool requires_grad) {
  if (requires_grad) {
    const DataType tensor_dtype = dtype();
    CHECK_OR_RETURN(IsFloatingDataType(tensor_dtype) || tensor_dtype == DataType::kBFloat16
                    || tensor_dtype == DataType::kFloat16)
        << "RuntimeError: only Tensors of floating point can require gradients";
  }
  autograd_meta_->set_requires_grad(requires_grad);
  return Maybe<void>::Ok();
}

Maybe<Tensor> TensorImpl::acc_grad() const { return autograd_meta_->acc_grad(); }

Maybe<TensorArg> TensorImpl::current_grad() const { return autograd_meta_->current_grad(); }

Maybe<void> TensorImpl::set_acc_grad(const std::shared_ptr<Tensor>& grad) {
  return autograd_meta_->set_acc_grad(grad);
}

Maybe<Tensor> TensorImpl::mut_acc_grad() { return autograd_meta_->mut_acc_grad(); }

Maybe<void> TensorImpl::set_retain_grad(bool retain_grad) {
  autograd_meta_->set_retain_grad(retain_grad);
  return Maybe<void>::Ok();
}

Maybe<LocalTensorImpl> LazyLocalTensorImpl::detach() const {
  auto detached_impl = std::make_shared<LazyLocalTensorImpl>(tensor_meta_, false, true);
  return std::shared_ptr<LocalTensorImpl>(detached_impl);
}

EagerLocalTensorImpl::EagerLocalTensorImpl()
    : LocalTensorImpl(std::make_shared<const LocalTensorMeta>(), false, false) {}

EagerLocalTensorImpl::EagerLocalTensorImpl(
    const std::shared_ptr<const LocalTensorMeta>& tensor_meta, bool requires_grad, bool is_leaf)
    : LocalTensorImpl(tensor_meta, requires_grad, is_leaf) {}

EagerLocalTensorImpl::~EagerLocalTensorImpl() {}

EagerLocalTensorImpl::EagerLocalTensorImpl(
    const std::shared_ptr<const LocalTensorMeta>& tensor_meta,
    const std::shared_ptr<TensorStorage>& tensor_storage, bool requires_grad, bool is_leaf)
    : LocalTensorImpl(tensor_meta, requires_grad, is_leaf), tensor_storage_(tensor_storage) {}

Maybe<void> EagerLocalTensorImpl::UpdateTensorStorage() {
  const auto& eager_blob_object = eager_blob_object_;
  tensor_storage_ = std::make_shared<TensorStorage>(eager_blob_object->tensor_storage());
  tensor_storage_->set_releaser_hook(
      [eager_blob_object](const std::shared_ptr<vm::TensorStorage>&) {
        CHECK_JUST(PhysicalRun([&](InstructionsBuilder* builder) -> Maybe<void> {
          if (eager_blob_object->producer_stream().has_value()) {
            JUST(builder->ReleaseTensor(eager_blob_object));
          }
          return Maybe<void>::Ok();
        }));
      });
  return Maybe<void>::Ok();
}

Maybe<LocalDepObject*> EagerLocalTensorImpl::compute_local_dep_object() const {
  return JUST(eager_blob_object())->compute_local_dep_object();
}

Maybe<void> EagerLocalTensorImpl::InitEagerBlobObject(
    const intrusive::shared_ptr<LocalDepObject>& dep_object) {
  CHECK_OR_RETURN(static_cast<bool>(device()));
  const auto& mem_case = device()->mem_case();
  const auto& mut_shape = std::const_pointer_cast<Shape>(tensor_meta()->shape_ptr());
  const auto& mut_stride = std::const_pointer_cast<Stride>(tensor_meta()->stride_ptr());

  if (tensor_storage_) {
    auto tensor_storage = tensor_storage_->storage();
    eager_blob_object_ = std::make_shared<vm::EagerBlobObject>(mem_case, mut_shape, mut_stride,
                                                               dtype(), tensor_storage, dep_object);
  } else {
    const auto& eager_blob_object =
        std::make_shared<vm::EagerBlobObject>(mem_case, mut_shape, mut_stride, dtype(),
                                              std::make_shared<vm::TensorStorage>(), dep_object);
    JUST(set_eager_blob_object(eager_blob_object));
  }
  return Maybe<void>::Ok();
}

Maybe<bool> EagerLocalTensorImpl::is_pinned() const {
  if (!eager_blob_object_) { return false; }
  return IsStreamAllocatorPinned::Visit(JUST(eager_blob_object_->producer_stream())->stream_role());
}

Maybe<void> EagerLocalTensorImpl::set_eager_blob_object(
    std::shared_ptr<vm::EagerBlobObject> eager_blob_object) {
  eager_blob_object_ = eager_blob_object;
  CHECK_OR_RETURN(eager_blob_object_->shape_ptr().get() == tensor_meta()->shape_ptr().get())
      << kOfBugIssueUploadPrompt;
  CHECK_OR_RETURN(eager_blob_object_->data_type() == tensor_meta()->dtype())
      << kOfBugIssueUploadPrompt;
  JUST(UpdateTensorStorage());
  return Maybe<void>::Ok();
}

std::shared_ptr<const Shape> EagerLocalTensorImpl::shape() const {
  if (!eager_blob_object_) { return tensor_meta()->shape_ptr(); }
  return eager_blob_object_->shape_ptr();
}

std::shared_ptr<const Stride> EagerLocalTensorImpl::stride() const {
  if (!eager_blob_object_) { return tensor_meta()->stride_ptr(); }
  return eager_blob_object_->stride_ptr();
  ;
}

Maybe<LocalTensorImpl> EagerLocalTensorImpl::detach() const {
  auto detached_impl =
      std::make_shared<EagerLocalTensorImpl>(tensor_meta_, tensor_storage_, false, true);
  detached_impl->eager_blob_object_ = eager_blob_object_;
  return std::shared_ptr<LocalTensorImpl>(detached_impl);
}

Maybe<void> EagerLocalTensorImpl::RegisterStorageDeleteHook(const std::function<void()>& hook) {
  CHECK_OR_RETURN(eager_blob_object_) << "EagerBlobObject has not initialized";
  eager_blob_object_->RegisterStorageDeleteHook(hook);
  return Maybe<void>::Ok();
}

Maybe<GlobalTensorImpl> LazyGlobalTensorImpl::detach() const {
  auto detached_impl = std::make_shared<LazyGlobalTensorImpl>(tensor_meta_, false, true);
  return std::shared_ptr<GlobalTensorImpl>(detached_impl);
}

EagerGlobalTensorImpl::EagerGlobalTensorImpl(
    Symbol<GlobalTensorMeta> global_tensor_meta, bool requires_grad, bool is_leaf,
    const std::shared_ptr<LocalTensor>& cur_rank_phy_tensor)
    : GlobalTensorImpl(global_tensor_meta, cur_rank_phy_tensor->requires_grad(),
                       cur_rank_phy_tensor->is_leaf()),
      cur_rank_phy_tensor_(cur_rank_phy_tensor) {}

/* static */ Maybe<EagerGlobalTensorImpl> EagerGlobalTensorImpl::New(
    Symbol<GlobalTensorMeta> global_tensor_meta, bool requires_grad, bool is_leaf) {
  const auto& parallel_desc = global_tensor_meta->parallel_desc();
  Optional<int64_t> parallel_id;
  const auto& device = JUST(parallel_desc->GetTensorDevice4CurrentProcessCtx(&parallel_id));
  return EagerGlobalTensorImpl::New(global_tensor_meta, device, parallel_id, requires_grad,
                                    is_leaf);
}

namespace {

Maybe<Shape> GetPhysicalShape(const Shape& logical_shape, const NdSbp& nd_sbp,
                              const ParallelDesc& parallel_desc,
                              const Optional<int64_t>& parallel_id) {
  if (parallel_id.has_value()) {
    return GetPhysicalShape(logical_shape, nd_sbp, parallel_desc, JUST(parallel_id));
  } else {
    return std::make_shared<Shape>(DimVector(logical_shape.NumAxes(), 0));
  }
}

}  // namespace

/* static */ Maybe<EagerGlobalTensorImpl> EagerGlobalTensorImpl::New(
    Symbol<GlobalTensorMeta> global_tensor_meta, Symbol<Device> device,
    const Optional<int64_t>& parallel_id, bool requires_grad, bool is_leaf) {
  const auto& shape = global_tensor_meta->shape_ptr();
  const auto& dtype = global_tensor_meta->dtype();
  const auto& nd_sbp = global_tensor_meta->nd_sbp();
  const auto& parallel_desc = global_tensor_meta->parallel_desc();
  const auto& cur_rank_phy_shape =
      JUST(GetPhysicalShape(*shape, *nd_sbp, *parallel_desc, parallel_id));
  std::shared_ptr<LocalTensor> cur_rank_phy_tensor;
  // If the `'parallel_desc` doesn't cover current ProcessCtx or the tensor has 0-size shape, there
  // is no need to compute through the corresponding opkernel, and can be obtained directly through
  // empty op.
  if (parallel_id.has_value() && shape->elem_cnt() != 0) {
    const auto& cur_rank_phy_tensor_meta =
        std::make_shared<LocalTensorMeta>(cur_rank_phy_shape, dtype, device);
    auto cur_rank_phy_tensor_impl =
        std::make_shared<EagerLocalTensorImpl>(cur_rank_phy_tensor_meta, requires_grad, is_leaf);
    const auto& dep_object = NewLocalDepObject();
    JUST(cur_rank_phy_tensor_impl->InitEagerBlobObject(dep_object));
    cur_rank_phy_tensor = std::make_shared<LocalTensor>(cur_rank_phy_tensor_impl);
  } else {
    const auto& dtype_symbol = JUST(DType::Get(dtype));
    const auto& empty =
        JUST(functional::Empty(*cur_rank_phy_shape, dtype_symbol, device, /*pin_memory=*/false));
    cur_rank_phy_tensor = JUST(empty->AsLocalTensor());
    JUST(cur_rank_phy_tensor->set_requires_grad(requires_grad));
    cur_rank_phy_tensor->set_is_leaf(is_leaf);
  }
  auto* tensor_impl =
      new EagerGlobalTensorImpl(global_tensor_meta, cur_rank_phy_tensor->requires_grad(),
                                cur_rank_phy_tensor->is_leaf(), cur_rank_phy_tensor);
  return std::shared_ptr<EagerGlobalTensorImpl>(tensor_impl);
}

Maybe<GlobalTensorImpl> EagerGlobalTensorImpl::detach() const {
  auto detached_impl = JUST(EagerGlobalTensorImpl::New(tensor_meta_, false, true));
  detached_impl->cur_rank_phy_tensor_ = cur_rank_phy_tensor_;
  detached_impl->consumer_nd_sbp_constraint_ = consumer_nd_sbp_constraint_;
  detached_impl->transport_token_ = transport_token_;
  return std::shared_ptr<GlobalTensorImpl>(detached_impl);
}

std::shared_ptr<const Stride> EagerGlobalTensorImpl::stride() const {
  if (!cur_rank_phy_tensor_) { return tensor_meta()->stride_ptr(); }
  const auto& stride_ptr = cur_rank_phy_tensor_->tensor_meta().stride_ptr();
  return stride_ptr;
}

}  // namespace one
}  // namespace oneflow
