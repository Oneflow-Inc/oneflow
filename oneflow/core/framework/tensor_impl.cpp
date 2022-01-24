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
#include "oneflow/core/common/spin_counter.h"
#include "oneflow/core/framework/instructions_builder.h"
#include "oneflow/core/framework/tensor_impl.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/stride.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/job/sbp_parallel.cfg.h"
#include "oneflow/core/functional/functional.h"
#include "oneflow/core/framework/device.h"
#include "oneflow/core/framework/dtype.h"
#include "oneflow/core/eager/eager_blob_object.h"
#include "oneflow/core/eager/local_dep_object.h"
#include "oneflow/core/vm/vm_util.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/control/global_process_ctx.h"
#include "oneflow/core/register/ofblob.h"

namespace oneflow {
namespace one {

Maybe<void> TensorImpl::set_requires_grad(bool requires_grad) {
  if (requires_grad) {
    const DataType tensor_dtype = dtype();
    CHECK_OR_RETURN(IsFloatingDataType(tensor_dtype) || tensor_dtype == DataType::kBFloat16
                    || tensor_dtype == DataType::kFloat16)
        << "RuntimeError: only Tensors of floating point can require gradients";
  }
  requires_grad_ = requires_grad;
  if (autograd_meta_) { autograd_meta_->set_requires_grad(requires_grad); }
  return Maybe<void>::Ok();
}

Maybe<Tensor> TensorImpl::acc_grad() const {
  CHECK_NOTNULL_OR_RETURN(autograd_meta_);
  return autograd_meta_->acc_grad();
}

Maybe<TensorArg> TensorImpl::current_grad() const {
  CHECK_NOTNULL_OR_RETURN(autograd_meta_);
  return autograd_meta_->current_grad();
}

Maybe<void> TensorImpl::set_acc_grad(const std::shared_ptr<Tensor>& grad) {
  CHECK_NOTNULL_OR_RETURN(autograd_meta_);
  return autograd_meta_->set_acc_grad(grad);
}

Maybe<Tensor> TensorImpl::mut_acc_grad() {
  CHECK_NOTNULL_OR_RETURN(autograd_meta_);
  return autograd_meta_->mut_acc_grad();
}

Maybe<void> TensorImpl::set_retain_grad(bool retain_grad) {
  CHECK_NOTNULL_OR_RETURN(autograd_meta_);
  autograd_meta_->set_retain_grad(retain_grad);
  return Maybe<void>::Ok();
}

Maybe<MirroredTensorImpl> LazyMirroredTensorImpl::detach() const {
  auto detached_impl = std::make_shared<LazyMirroredTensorImpl>(tensor_meta_, false, true);
  return std::shared_ptr<MirroredTensorImpl>(detached_impl);
}

EagerMirroredTensorImpl::EagerMirroredTensorImpl()
    : MirroredTensorImpl(std::make_shared<const MirroredTensorMeta>(), false, false) {}

EagerMirroredTensorImpl::EagerMirroredTensorImpl(
    const std::shared_ptr<const MirroredTensorMeta>& tensor_meta, bool requires_grad, bool is_leaf)
    : MirroredTensorImpl(tensor_meta, requires_grad, is_leaf) {}

EagerMirroredTensorImpl::~EagerMirroredTensorImpl() {}

EagerMirroredTensorImpl::EagerMirroredTensorImpl(
    const std::shared_ptr<const MirroredTensorMeta>& tensor_meta,
    const std::shared_ptr<TensorStorage>& tensor_storage, bool requires_grad, bool is_leaf)
    : MirroredTensorImpl(tensor_meta, requires_grad, is_leaf), tensor_storage_(tensor_storage) {}

Maybe<void> EagerMirroredTensorImpl::UpdateTensorStorage() {
  const auto& eager_blob_object = eager_blob_object_;
  tensor_storage_ = std::make_shared<TensorStorage>(eager_blob_object->tensor_storage());
  const auto& parallel_desc = JUST(Placement4Device(this->device())).shared_from_symbol();
  tensor_storage_->set_releaser_hook(
      [eager_blob_object, parallel_desc](const std::shared_ptr<vm::TensorStorage>&) {
        CHECK_JUST(PhysicalRun([&](InstructionsBuilder* builder) -> Maybe<void> {
          if (eager_blob_object->producer_op_device().has_value()) {
            JUST(builder->ReleaseTensor(eager_blob_object, parallel_desc));
            const auto& device = JUST(eager_blob_object->producer_op_device());
            auto* local_dep_object = JUST(eager_blob_object->compute_local_dep_object());
            JUST(PutLocalDepObjectToDevicePool(device, local_dep_object));
          }
          return Maybe<void>::Ok();
        }));
      });
  return Maybe<void>::Ok();
}

Maybe<LocalDepObject*> EagerMirroredTensorImpl::compute_local_dep_object() const {
  return JUST(eager_blob_object())->compute_local_dep_object();
}

Maybe<void> EagerMirroredTensorImpl::InitEagerBlobObject(LocalDepObject* dep_object) {
  CHECK_OR_RETURN(static_cast<bool>(device()));
  const auto& mem_case = device()->mem_case();
  const auto& mut_shape = std::const_pointer_cast<Shape>(tensor_meta()->shape_ptr());

  if (tensor_storage_) {
    auto tensor_storage = tensor_storage_->storage();
    eager_blob_object_ = std::make_shared<vm::EagerBlobObject>(mem_case, mut_shape, dtype(),
                                                               tensor_storage, dep_object);
  } else {
    const auto& eager_blob_object = std::make_shared<vm::EagerBlobObject>(
        mem_case, mut_shape, dtype(), std::make_shared<vm::TensorStorage>(), dep_object);
    JUST(set_eager_blob_object(eager_blob_object));
  }
  eager_blob_object_->set_storage_offset(tensor_meta()->storage_offset());
  return Maybe<void>::Ok();
}

Maybe<void> EagerMirroredTensorImpl::set_eager_blob_object(
    std::shared_ptr<vm::EagerBlobObject> eager_blob_object) {
  eager_blob_object_ = eager_blob_object;
  CHECK_OR_RETURN(eager_blob_object_->blob_desc().shape_ptr().get()
                  == tensor_meta()->shape_ptr().get());
  CHECK_OR_RETURN(eager_blob_object_->blob_desc().data_type() == tensor_meta()->dtype());
  JUST(UpdateTensorStorage());
  return Maybe<void>::Ok();
}

const std::shared_ptr<const Shape>& EagerMirroredTensorImpl::shape() const {
  if (!eager_blob_object_) { return tensor_meta()->shape_ptr(); }
  if (eager_blob_object_->is_shape_synced()) { return eager_blob_object_->blob_desc().shape_ptr(); }

  const auto& shape_ptr = eager_blob_object_->blob_desc().shape_ptr();
  const auto& Callback =
      std::make_shared<std::function<void(uint64_t)>>([&shape_ptr](uint64_t of_blob_ptr) {
        const auto* of_blob = reinterpret_cast<OfBlob*>(of_blob_ptr);
        of_blob->blob().shape_view().ToShape(const_cast<Shape*>(shape_ptr.get()));
      });
  CHECK_JUST(SpinCounter::SpinWait(1, [&](const std::shared_ptr<SpinCounter>& sc) -> Maybe<void> {
    return PhysicalRun([&](InstructionsBuilder* builder) -> Maybe<void> {
      return builder->SyncAccessBlobByCallback(this, sc, Callback, "const");
    });
  }));
  eager_blob_object_->set_is_shape_synced(true);
  return shape_ptr;
}

Maybe<MirroredTensorImpl> EagerMirroredTensorImpl::detach() const {
  auto detached_impl =
      std::make_shared<EagerMirroredTensorImpl>(tensor_meta_, tensor_storage_, false, true);
  detached_impl->eager_blob_object_ = eager_blob_object_;
  return std::shared_ptr<MirroredTensorImpl>(detached_impl);
}

MirroredTensorMeta::MirroredTensorMeta()
    : TensorMeta(std::make_shared<const Shape>(), DataType::kInvalidDataType),
      device_(Symbol<Device>()),
      stride_(std::make_shared<const Stride>()),
      storage_offset_(0) {}

MirroredTensorMeta::MirroredTensorMeta(const std::shared_ptr<const Shape>& shape, DataType dtype,
                                       Symbol<Device> device)
    : TensorMeta(shape, dtype),
      device_(device),
      stride_(std::make_shared<const Stride>(*shape)),
      storage_offset_(0) {}

MirroredTensorMeta::MirroredTensorMeta(const std::shared_ptr<const Shape>& shape, DataType dtype,
                                       Symbol<Device> device,
                                       const std::shared_ptr<const Stride>& stride,
                                       int64_t storage_offset)
    : TensorMeta(shape, dtype), device_(device), stride_(stride), storage_offset_(storage_offset) {}

bool MirroredTensorMeta::operator==(const MirroredTensorMeta& other) const {
  // It's correct to ignore is_dynamic_ field.
  return *this->shape_ptr() == *other.shape_ptr() && this->dtype() == other.dtype()
         && *this->device() == *other.device() && this->stride() == other.stride();
}

size_t MirroredTensorMeta::CalcHashValue() const {
  // It's correct to ignore is_dynamic_ field.
  return std::hash<Shape>()(*shape_ptr()) ^ std::hash<DataType>()(dtype())
         ^ std::hash<Device>()(*device()) ^ std::hash<Stride>()(stride());
}

bool ConsistentTensorMeta::operator==(const ConsistentTensorMeta& other) const {
  // It's correct to ignore is_dynamic_ field.
  return *this->shape_ptr() == *other.shape_ptr() && this->dtype() == other.dtype()
         && this->nd_sbp() == other.nd_sbp() && this->parallel_desc() == other.parallel_desc();
}

size_t ConsistentTensorMeta::CalcHashValue() const {
  return std::hash<Shape>()(*shape_ptr()) ^ std::hash<DataType>()(dtype())
         ^ std::hash<Symbol<cfg::NdSbp>>()(nd_sbp())
         ^ std::hash<Symbol<ParallelDesc>>()(parallel_desc());
}

Maybe<ConsistentTensorImpl> LazyConsistentTensorImpl::detach() const {
  auto detached_impl = std::make_shared<LazyConsistentTensorImpl>(tensor_meta_, false, true);
  return std::shared_ptr<ConsistentTensorImpl>(detached_impl);
}

EagerConsistentTensorImpl::EagerConsistentTensorImpl(
    Symbol<ConsistentTensorMeta> consistent_tensor_meta, bool requires_grad, bool is_leaf,
    const std::shared_ptr<MirroredTensor>& cur_rank_phy_tensor)
    : ConsistentTensorImpl(consistent_tensor_meta, cur_rank_phy_tensor->requires_grad(),
                           cur_rank_phy_tensor->is_leaf()),
      cur_rank_phy_tensor_(cur_rank_phy_tensor) {}

/* static */ Maybe<EagerConsistentTensorImpl> EagerConsistentTensorImpl::New(
    Symbol<ConsistentTensorMeta> consistent_tensor_meta, bool requires_grad, bool is_leaf) {
  const auto& parallel_desc = consistent_tensor_meta->parallel_desc();
  Optional<int64_t> parallel_id;
  const auto& device = JUST(parallel_desc->GetTensorDevice4CurrentProcessCtx(&parallel_id));
  return EagerConsistentTensorImpl::New(consistent_tensor_meta, device, parallel_id, requires_grad,
                                        is_leaf);
}

namespace {

Maybe<Shape> GetPhysicalShape(const Shape& logical_shape, const cfg::NdSbp& nd_sbp,
                              const ParallelDesc& parallel_desc,
                              const Optional<int64_t>& parallel_id) {
  if (parallel_id.has_value()) {
    return GetPhysicalShape(logical_shape, nd_sbp, parallel_desc, JUST(parallel_id));
  } else {
    return std::make_shared<Shape>(DimVector(logical_shape.NumAxes(), 0));
  }
}

}  // namespace

/* static */ Maybe<EagerConsistentTensorImpl> EagerConsistentTensorImpl::New(
    Symbol<ConsistentTensorMeta> consistent_tensor_meta, Symbol<Device> device,
    const Optional<int64_t>& parallel_id, bool requires_grad, bool is_leaf) {
  const auto& shape = consistent_tensor_meta->shape_ptr();
  const auto& dtype = consistent_tensor_meta->dtype();
  const auto& nd_sbp = consistent_tensor_meta->nd_sbp();
  const auto& parallel_desc = consistent_tensor_meta->parallel_desc();
  const auto& cur_rank_phy_shape =
      JUST(GetPhysicalShape(*shape, *nd_sbp, *parallel_desc, parallel_id));
  std::shared_ptr<MirroredTensor> cur_rank_phy_tensor;
  if (parallel_id.has_value()) {
    const auto& cur_rank_phy_tensor_meta =
        std::make_shared<MirroredTensorMeta>(cur_rank_phy_shape, dtype, device);
    auto cur_rank_phy_tensor_impl =
        std::make_shared<EagerMirroredTensorImpl>(cur_rank_phy_tensor_meta, requires_grad, is_leaf);
    const auto& dep_object = JUST(GetLocalDepObjectFromDevicePool(device));
    JUST(cur_rank_phy_tensor_impl->InitEagerBlobObject(dep_object));
    cur_rank_phy_tensor = std::make_shared<MirroredTensor>(cur_rank_phy_tensor_impl);
  } else {
    const auto& dtype_symbol = JUST(DType::Get(dtype));
    const auto& empty = JUST(functional::Empty(*cur_rank_phy_shape, dtype_symbol, device));
    cur_rank_phy_tensor = JUST(empty->AsMirroredTensor());
    JUST(cur_rank_phy_tensor->set_requires_grad(requires_grad));
    cur_rank_phy_tensor->set_is_leaf(is_leaf);
  }
  auto* tensor_impl =
      new EagerConsistentTensorImpl(consistent_tensor_meta, cur_rank_phy_tensor->requires_grad(),
                                    cur_rank_phy_tensor->is_leaf(), cur_rank_phy_tensor);
  return std::shared_ptr<EagerConsistentTensorImpl>(tensor_impl);
}

Maybe<ConsistentTensorImpl> EagerConsistentTensorImpl::detach() const {
  auto detached_impl = JUST(EagerConsistentTensorImpl::New(tensor_meta_, false, true));
  detached_impl->cur_rank_phy_tensor_ = cur_rank_phy_tensor_;
  detached_impl->consumer_nd_sbp_constraint_ = consumer_nd_sbp_constraint_;
  detached_impl->transport_token_ = transport_token_;
  return std::shared_ptr<ConsistentTensorImpl>(detached_impl);
}

}  // namespace one
}  // namespace oneflow
