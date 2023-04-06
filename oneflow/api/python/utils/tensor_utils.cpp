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
#include "oneflow/api/python/utils/tensor_utils.h"

#include "oneflow/core/autograd/autograd_engine.h"
#include "oneflow/core/common/container_util.h"
#include "oneflow/core/common/switch_func.h"
#include "oneflow/core/common/tensor_buffer.h"
#include "oneflow/core/framework/nd_sbp.h"
#include "oneflow/core/functional/functional.h"
#include "oneflow/core/job/global_mode.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/extension/python/numpy.h"
#include "oneflow/core/common/decorator.h"
#include "oneflow/core/framework/consistency_check.h"
#include "oneflow/core/functional/impl/common.h"

namespace py = pybind11;

namespace oneflow {
namespace one {

Maybe<void> EagerLocalTensorZeros(const std::shared_ptr<Tensor>& t) {
  JUST(functional::CheckInplaceValid(t));
  std::shared_ptr<LocalTensor> local_tensor;
  if (t->is_local()) {
    local_tensor = JUST(t->AsLocalTensor());
  } else {
    local_tensor = JUST(t->cur_rank_phy_tensor());
  }
  CHECK_OR_RETURN(local_tensor->is_eager()) << "eager tensors supported only";
  JUST(PhysicalRun([&](InstructionsBuilder* builder) -> Maybe<void> {
    JUST(builder->AccessBlobByCallback(
        local_tensor,
        [](ep::Stream* stream, const std::shared_ptr<vm::EagerBlobObject>& eager_blob_object) {
          AutoMemset(stream, eager_blob_object->mut_dptr(), 0,
                     eager_blob_object->ByteSizeOfBlobBody(), eager_blob_object->mem_case());
        },
        "mut"));
    return Maybe<void>::Ok();
  }));
  return Maybe<void>::Ok();
}

namespace {
void CopyFromNumpyArray(ep::Stream* stream,
                        const std::shared_ptr<vm::EagerBlobObject>& eager_blob_object,
                        const NumPyArrayPtr& array_ptr) {
  SyncAutoMemcpy(stream, eager_blob_object->mut_dptr(), array_ptr.data(),
                 eager_blob_object->ByteSizeOfBlobBody(), eager_blob_object->mem_case(),
                 memory::MakeHostMemCase());
}
}  // namespace

Maybe<void> CopyLocalTensorFromUntypedArray(const std::shared_ptr<Tensor>& tensor,
                                            PyObject* array) {
  return CopyBetweenLocalTensorAndNumpy(tensor, array, CopyFromNumpyArray, "mut",
                                        /*block_host_until_done=*/false);
}

Maybe<std::tuple<std::vector<Shape>, std::vector<Symbol<DType>>>>
MaybeGetTensorBufferShapesAndDTypes(const std::shared_ptr<Tensor>& t) {
  const auto& tensor = JUST(t->AsLocalTensor());
  if (tensor->dtype() != DType::TensorBuffer()) {
    return Error::RuntimeError() << "tensor buffer supported only";
  }
  CHECK_OR_RETURN(tensor->is_eager()) << "eager tensors supported only";
  std::vector<Shape> shapes;
  std::vector<Symbol<DType>> dtypes;

  auto btb = std::make_shared<BlockingThenBusy>();
  JUST(PhysicalRun([&](InstructionsBuilder* builder) -> Maybe<void> {
    return builder->SyncAccessBlobByCallback(
        tensor, btb, [](ep::Stream* stream, const std::shared_ptr<vm::EagerBlobObject>&) {},
        "const");
  }));
  JUST(btb->WaitUntilCntEqualZero(VirtualMachine::GetPredicatorNoMoreInstructionsFinished()));

  const auto& eager_blob_object = JUST(tensor->eager_blob_object());
  const Shape& blob_shape = eager_blob_object->shape();
  const auto* tensor_buffer_ptr = eager_blob_object->dptr<TensorBuffer>();
  for (int64_t i = 0; i < blob_shape.elem_cnt(); ++i) {
    const TensorBuffer* tensor_buffer = tensor_buffer_ptr + i;
    shapes.emplace_back(tensor_buffer->shape());
    dtypes.emplace_back(DType::Get(tensor_buffer->data_type()).GetOrThrow());
  }
  return std::make_tuple(shapes, dtypes);
}

Maybe<void> RegisterTensorHook(const std::shared_ptr<Tensor>& self,
                               const AutogradMeta::Hook& hook) {
  CHECK_OR_RETURN(self->requires_grad())
      << "cannot register a hook on a tensor that doesn't require gradient";
  if (!self->grad_fn_node()) { JUST(AddAccumulateFunctionNode(self)); }
  self->mut_autograd_meta()->add_hook(hook);
  return Maybe<void>::Ok();
}

Maybe<void> RegisterTensorPostGradAccumulationHook(const std::shared_ptr<Tensor>& self,
                                                   const AutogradMeta::Hook& hook) {
  if (!self->grad_fn_node()) { JUST(AddAccumulateFunctionNode(self)); }
  self->mut_autograd_meta()->add_post_grad_accumulation_hook(hook);
  return Maybe<void>::Ok();
}

Maybe<py::tuple> TensorGetPyTupleOfSbp(const Tensor& tensor) {
  const auto& nd_sbp = JUST(tensor.nd_sbp());
  const auto& tuple = std::make_shared<py::tuple>(nd_sbp->sbp_parallel_size());
  for (int i = 0; i < nd_sbp->sbp_parallel_size(); ++i) {
    (*tuple)[i] = SymbolOf(nd_sbp->sbp_parallel(i));
  }
  return tuple;
}

Maybe<Tensor> MakeLocalTensorFromData(PyObject* data, const Optional<Symbol<DType>>& dtype,
                                      const Optional<Symbol<Device>>& device,
                                      const bool requires_grad, const bool pin_memory) {
  bool is_bfloat16_dtype = dtype ? JUST(dtype)->data_type() == DataType::kBFloat16 : false;
  bool is_cuda_device = device ? JUST(device)->enum_type() == DeviceType::kCUDA : false;
  if (is_bfloat16_dtype && is_cuda_device) {
#if CUDA_VERSION < 11000
    return Error::RuntimeError()
           << "Cannot create a bfloat16 tensor on gpu under cuda version: 11000";
#endif  // CUDA_VERSION >= 11000
  }
  PyArray_Descr* np_dtype =
      dtype.has_value() && !is_bfloat16_dtype
          ? PyArray_DescrFromType(JUST(numpy::OFDataTypeToNumpyType(JUST(dtype)->data_type())))
          : nullptr;
  // NPY_ARRAY_DEFAULT is NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_BEHAVED, so the
  // array with NPY_ARRAY_DEFAULT flag is C-style contiguous.
  // NPY_ARRAY_FORCECAST is needed otherwise there will a segfault.
  //
  // Even though PyArray_FromAny can cast the input array to the desired dtype
  // if `dtype` argument is set, it fails to handle the following case:
  // >> x = [flow.tensor([1, 2])] * 3 <-- x is a list of flow.Tensor
  // >> y = flow.tensor(x, dtype=flow.float32) <-- returns nullptr
  // However, the following case without `dtype` argument works well:
  // >> x = [flow.tensor([1, 2])] * 3
  // >> y = flow.tensor(x)
  // So we cast the input array to the desired dtype manually.
  PyArrayObject* _array = reinterpret_cast<PyArrayObject*>(
      PyArray_FromAny(data, nullptr, 0, 0,
                      NPY_ARRAY_DEFAULT | NPY_ARRAY_ENSURECOPY | NPY_ARRAY_FORCECAST, nullptr));
  if (!_array) {
    return Error::RuntimeError() << "Can not convert input data to a new numpy array.";
  }
  // PyArray_FromArray steals a reference to np_dtype object, so no need to decref it.
  PyObject* array = PyArray_FromArray(
      _array, np_dtype, NPY_ARRAY_DEFAULT | NPY_ARRAY_ENSURECOPY | NPY_ARRAY_FORCECAST);
  Py_DECREF(_array);
  auto* np_arr = reinterpret_cast<PyArrayObject*>(array);
  const npy_intp* dims_ptr = PyArray_SHAPE(np_arr);
  const Shape shape(DimVector(dims_ptr, dims_ptr + PyArray_NDIM(np_arr)));
  DataType np_data_type = JUST(numpy::GetOFDataTypeFromNpArray(np_arr));

  Symbol<Device> device_;
  if (device) {
    device_ = JUST(device);
  } else {
    device_ = JUST(Device::New("cpu"));
  }
  std::shared_ptr<Tensor> tensor =
      JUST(functional::Empty(shape, JUST(DType::Get(np_data_type)), device_,
                             /*requires_grad=*/false, /*pin_memory=*/pin_memory));
  JUST(CopyLocalTensorFromUntypedArray(tensor, array));

  Py_DECREF(array);
  if (dtype && JUST(dtype)->data_type() != np_data_type) {
    tensor = JUST(functional::To(tensor, JUST(dtype), false));
  } else if (!dtype && !PyArray_Check(data) && tensor->dtype()->is_floating_point()
             && GetDefaultDType() != tensor->dtype()) {
    // If it not assign dtype and created from PySequence, cast tensor to default floating dtype
    tensor = JUST(functional::To(tensor, JUST(DType::Get(DataType::kFloat)), false));
  }
  JUST(tensor->set_requires_grad(requires_grad));
  return tensor;
}

namespace {

Maybe<Symbol<NdSbp>> GetAllBroadcastNdSbp(size_t ndim) {
  NdSbp broadcast_nd_sbp;
  for (size_t i = 0; i < ndim; ++i) {
    broadcast_nd_sbp.mutable_sbp_parallel()->Add()->mutable_broadcast_parallel();
  }
  return SymbolOf(broadcast_nd_sbp);
}

auto* CachedGetAllBroadcastNdSbp = DECORATE(&GetAllBroadcastNdSbp, ThreadLocal);

}  // namespace

Maybe<Tensor> MakeGlobalTensorFromData(PyObject* data, const Optional<Symbol<DType>>& dtype,
                                       Symbol<ParallelDesc> placement,
                                       const std::vector<Symbol<SbpParallel>>& sbp_tuple,
                                       const bool requires_grad) {
  PyObject* array = NULL;
  if (PyArray_Check(data)) {
    // Only NPY_CORDER is supported, and returns a new C-style contiguous array.
    array = PyArray_NewCopy((PyArrayObject*)data, NPY_CORDER);
  } else {
    // NPY_ARRAY_DEFAULT is NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_BEHAVED, so the
    // array with NPY_ARRAY_DEFAULT flag is C-style contiguous.
    array = PyArray_FromAny(data, nullptr, 0, 0, NPY_ARRAY_DEFAULT | NPY_ARRAY_ENSURECOPY, nullptr);
    if (!array) { return Error::RuntimeError() << "Can not convert input data to a numpy array."; }
  }
  auto* np_arr = reinterpret_cast<PyArrayObject*>(array);
  const npy_intp* dims_ptr = PyArray_SHAPE(np_arr);
  const Shape shape(DimVector(dims_ptr, dims_ptr + PyArray_NDIM(np_arr)));
  DataType data_type = JUST(numpy::GetOFDataTypeFromNpArray(np_arr));

  if (placement->parallel_num() > 1) {
    const void* buf_ptr = PyArray_DATA(np_arr);
    size_t array_size = PyArray_SIZE(np_arr);
    CHECK_EQ_OR_RETURN(array_size, shape.elem_cnt());
    size_t byte_size = array_size * GetSizeOfDataType(data_type);
    JUST(DataConsistencyCheck(buf_ptr, byte_size, placement));
  }

  Symbol<Device> device = JUST(Device::New(placement->device_tag()));
  std::shared_ptr<Tensor> local_tensor;
  {
    GlobalMode::Guard guard(/* disable global mode */ false);
    local_tensor =
        JUST(functional::Empty(shape, JUST(DType::Get(data_type)), device, /*requires_grad=*/false,
                               /*pin_memory=*/false));
  }
  JUST(CopyLocalTensorFromUntypedArray(local_tensor, array));

  Py_DECREF(array);
  // Cast to float if data is double sequence, rather than numpy array.
  Symbol<DType> dtype_;
  if (dtype) {
    dtype_ = JUST(dtype);
  } else if (!dtype && data_type == DataType::kDouble && !PyArray_Check(data)) {
    dtype_ = DType::Float();
  }
  if (dtype_) { local_tensor = JUST(functional::Cast(local_tensor, dtype_, /*pin_memory=*/false)); }

  size_t sbp_dims = sbp_tuple.size();
  Symbol<NdSbp> broadcast_nd_sbp = JUST(CachedGetAllBroadcastNdSbp(sbp_dims));

  std::shared_ptr<Tensor> broadcast_tensor = JUST(
      functional::LocalToGlobal(local_tensor, placement, *JUST(GetSbpList(broadcast_nd_sbp)), shape,
                                local_tensor->dtype(), /* sync_data */ true, /*copy=*/false));

  std::vector<Symbol<SbpParallel>> grad_sbp_tuple;
  auto global_tensor =
      JUST(functional::ToGlobal(broadcast_tensor, placement, sbp_tuple, grad_sbp_tuple,
                                /* check_meta */ false, /*copy=*/false));
  JUST(global_tensor->set_requires_grad(requires_grad));
  return global_tensor;
}

Maybe<Tensor> MakeTensorFromOtherTensor(const std::shared_ptr<Tensor>& other,
                                        const bool pin_memory) {
  if (other->is_local()) {
    const Symbol<Device>& device = JUST(other->device());
    return functional::Copy(other, device->type(), device->device_id(), pin_memory);
  } else {
    const Symbol<NdSbp>& nd_sbp = JUST(other->nd_sbp());
    const std::vector<Symbol<SbpParallel>>& sbp_tuple = *JUST(GetSbpList(nd_sbp));
    std::vector<Symbol<SbpParallel>> grad_sbp_tuple;
    // TODO:(zhaoluyang) global case support pin_memory
    return functional::ToGlobal(other, JUST(other->parallel_desc()), sbp_tuple, grad_sbp_tuple,
                                /* check_meta */ false, /*copy=*/false);
  }
}

Maybe<Tensor> MakeTensorFromOtherTensor(const std::shared_ptr<Tensor>& other,
                                        const Optional<Symbol<DType>>& dtype,
                                        const Optional<Symbol<Device>>& device,
                                        const bool requires_grad, const bool pin_memory) {
  std::shared_ptr<Tensor> tensor;
  Symbol<Device> device_;
  if (device) { device_ = JUST(device); }
  if (other->is_local()) {
    if (!device) { device_ = JUST(other->device()); }
    tensor = JUST(functional::Copy(other, device_->type(), device_->device_id(),
                                   pin_memory && !dtype.has_value()));
  } else {
    tensor = JUST(functional::GlobalToLocal(other, /*copy=*/false));
    if (!device) { device_ = JUST(Device::New("cpu")); }
    tensor = JUST(functional::Copy(tensor, device_->type(), device_->device_id(),
                                   pin_memory && !dtype.has_value()));
  }
  if (dtype) {
    const Symbol<DType>& dtype_ = JUST(dtype);
    if (tensor->dtype() != dtype_) { tensor = JUST(functional::Cast(tensor, dtype_, pin_memory)); }
  }
  JUST(tensor->set_requires_grad(requires_grad));
  return tensor;
}

Maybe<Tensor> MakeTensorFromOtherTensor(const std::shared_ptr<Tensor>& other,
                                        const Optional<Symbol<DType>>& dtype,
                                        const Symbol<ParallelDesc>& placement,
                                        const std::vector<Symbol<SbpParallel>>& sbp_tuple,
                                        const bool requires_grad) {
  std::vector<Symbol<SbpParallel>> grad_sbp_tuple;
  bool check_meta = other->is_global() ? false : true;
  std::shared_ptr<Tensor> tensor = JUST(functional::ToGlobal(
      other, placement, sbp_tuple, grad_sbp_tuple, check_meta, /*copy=*/false));
  if (dtype) {
    const Symbol<DType>& dtype_ = JUST(dtype);
    if (tensor->dtype() != dtype_) {
      tensor = JUST(functional::Cast(tensor, dtype_, /*pin_memory=*/false));
    }
  }
  JUST(tensor->set_requires_grad(requires_grad));
  return tensor;
}

}  // namespace one
}  // namespace oneflow
