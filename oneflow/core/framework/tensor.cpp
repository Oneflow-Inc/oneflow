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
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/framework/device.h"
#include "oneflow/core/framework/dtype.h"
#include "oneflow/core/autograd/autograd_engine.h"
#include "oneflow/core/framework/op_interpreter/eager_mirrored_op_interpreter.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/framework/op_expr.h"

namespace oneflow {

namespace one {

/*static*/ Maybe<MirroredTensor> MirroredTensor::MakeTensor(
    const std::shared_ptr<const Shape>& shape, DataType dtype,
    const std::shared_ptr<const Device>& device, bool is_lazy, bool requires_grad, bool is_leaf) {
  std::shared_ptr<MirroredTensorImpl> impl;
  if (is_lazy) {
    impl = std::make_shared<LazyMirroredTensorImpl>(shape, dtype, device, requires_grad, is_leaf);
  } else {
    const auto eager_blob_object =
        CHECK_JUST(GenerateAllocatedEagerBlobObject(dtype, *shape, device));
    impl = std::make_shared<EagerMirroredTensorImpl>(eager_blob_object, device, requires_grad,
                                                     is_leaf);
  }
  return std::make_shared<MirroredTensor>(impl);
}

/*static*/ std::shared_ptr<MirroredTensor> MirroredTensor::MakeEagerTensor(
    const std::shared_ptr<vm::EagerBlobObject> eager_blob_object,
    const std::shared_ptr<const Device>& device, bool requires_grad, bool is_leaf) {
  std::shared_ptr<MirroredTensorImpl> impl =
      std::make_shared<EagerMirroredTensorImpl>(eager_blob_object, device, requires_grad, is_leaf);
  return std::make_shared<MirroredTensor>(impl);
}

/*static*/ std::shared_ptr<MirroredTensor> MirroredTensor::MakeEagerTensor(
    const std::shared_ptr<vm::EagerBlobObject> eager_blob_object,
    const std::shared_ptr<const Device>& device,
    const std::shared_ptr<TensorStorage> tensor_storage, bool requires_grad, bool is_leaf) {
  std::shared_ptr<MirroredTensorImpl> impl = std::make_shared<EagerMirroredTensorImpl>(
      eager_blob_object, device, tensor_storage, requires_grad, is_leaf);
  return std::make_shared<MirroredTensor>(impl);
}

bool MirroredTensor::is_cuda() const { return CHECK_JUST(device())->type() == "cuda"; }

int64_t MirroredTensor::ndim() const { return shape()->NumAxes(); }

int64_t MirroredTensor::dim(int64_t index) const { return shape()->At(index); }

int64_t MirroredTensor::nelement() const { return shape()->elem_cnt(); }

std::shared_ptr<MirroredTensor> MirroredTensor::data() const {
  std::shared_ptr<MirroredTensor> t = std::make_shared<MirroredTensor>(impl_);
  return t;
}

Maybe<MirroredTensor> MirroredTensor::api_detach() const {
  const auto& eager_blob_object = JUST(impl_->eager_blob_object());
  const auto& device = impl_->device();
  const auto& tensor_storage = JUST(this->tensor_storage());
  std::shared_ptr<MirroredTensor> t =
      MirroredTensor::MakeEagerTensor(eager_blob_object, device, tensor_storage, false, true);
  return t;
}

Maybe<Tensor> MirroredTensor::clone() const {
  const auto& device_type = JUST(this->device())->type();
  int64_t device_id = JUST(this->device())->device_id();
  std::shared_ptr<OpExpr> copy_op_ = JUST(one::OpBuilder("copy")
                                              .Input("in", 1)
                                              .Attr("device_type", device_type)
                                              .Attr("device_id", device_id)
                                              .Output("out", 1)
                                              .Build());
  std::shared_ptr<MirroredTensor> input =
      std::const_pointer_cast<MirroredTensor>(shared_from_this());
  const auto& output = JUST(OpInterpUtil::Dispatch<Tensor>(*copy_op_, {input}));
  return output;
}

Maybe<ConsistentTensor> ConsistentTensor::MakeTensor(
    const std::shared_ptr<const Shape>& shape, DataType dtype,
    Symbol<cfg::ParallelDistribution> parallel_distribution, Symbol<ParallelDesc> parallel_desc,
    bool is_lazy, bool requires_grad, bool is_leaf) {
  std::shared_ptr<ConsistentTensorImpl> impl;
  Symbol<ConsistentTensorMeta> consistent_tensor_meta(
      ConsistentTensorMeta(shape, dtype, parallel_distribution, parallel_desc));
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

int64_t ConsistentTensor::dim(int64_t index) const { return shape()->At(index); }

int64_t ConsistentTensor::nelement() const { return shape()->elem_cnt(); }

int64_t ConsistentTensor::ndim() const { return shape()->NumAxes(); }

std::shared_ptr<ConsistentTensor> ConsistentTensor::data() const {
  std::shared_ptr<ConsistentTensor> t = std::make_shared<ConsistentTensor>(impl_);
  return t;
}

Maybe<ConsistentTensor> ConsistentTensor::api_detach() const {
  std::shared_ptr<ConsistentTensor> t = std::make_shared<ConsistentTensor>(impl_);
  return t;
}

}  // namespace one

}  // namespace oneflow
