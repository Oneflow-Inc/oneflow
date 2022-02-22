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

#include "oneflow/api/python/ofblob/ofblob.e.h"
#include "oneflow/core/autograd/autograd_engine.h"
#include "oneflow/core/common/container_util.h"
#include "oneflow/core/common/switch_func.h"
#include "oneflow/core/common/tensor_buffer.h"
#include "oneflow/core/framework/nd_sbp.h"
#include "oneflow/core/functional/functional.h"
#include "oneflow/extension/python/numpy.h"
#include "oneflow/core/common/decorator.h"
#include "oneflow/core/framework/consistency_check.h"

namespace py = pybind11;

namespace oneflow {
namespace one {

Maybe<void> EagerMirroredTensorZeros(const std::shared_ptr<Tensor>& t) {
  std::shared_ptr<MirroredTensor> local_tensor;
  if (t->is_local()) {
    local_tensor = JUST(t->AsMirroredTensor());
  } else {
    local_tensor = JUST(t->cur_rank_phy_tensor());
  }
  CHECK_OR_RETURN(local_tensor->is_eager()) << "eager tensors supported only";
  JUST(PhysicalRun([&](InstructionsBuilder* builder) -> Maybe<void> {
    JUST(builder->AccessBlobByCallback(
        local_tensor,
        [](uint64_t of_blob_ptr) {
          auto* of_blob = reinterpret_cast<OfBlob*>(of_blob_ptr);
          of_blob->AsyncAutoMemset(0);
        },
        "mut"));
    return Maybe<void>::Ok();
  }));
  return Maybe<void>::Ok();
}

template<typename T>
Maybe<void> CopyMirroredTensorFromUntypedArray(const std::shared_ptr<Tensor>& tensor,
                                               PyObject* array) {
  return CopyBetweenMirroredTensorAndNumpy<T>(tensor, array, BlobNumpyCopyUtil<T>::From, "mut",
                                              /*block_host_until_done=*/false);
}

Maybe<std::string> GetCopyMirroredTensorToNumpyFuncName(DataType dtype) {
  using namespace oneflow;
  static const HashMap<int64_t, std::shared_ptr<std::string>> data_type2func_name{
#define DATA_TYPE_FUNC_NAME_PAIR(type_cpp, type_proto) \
  {type_proto, std::make_shared<std::string>("_copy_to_numpy_" #type_cpp)},
      OF_PP_FOR_EACH_TUPLE(DATA_TYPE_FUNC_NAME_PAIR, POD_DATA_TYPE_SEQ)
#undef DATA_TYPE_FUNC_NAME_PAIR
  };
  return JUST(MapAt(data_type2func_name, static_cast<int64_t>(dtype)));
}

Maybe<std::string> GetCopyMirroredTensorFromNumpyFuncName(DataType dtype) {
  using namespace oneflow;
  static const HashMap<int64_t, std::shared_ptr<std::string>> data_type2func_name{
#define DATA_TYPE_FUNC_NAME_PAIR(type_cpp, type_proto) \
  {type_proto, std::make_shared<std::string>("_copy_from_numpy_" #type_cpp)},
      OF_PP_FOR_EACH_TUPLE(DATA_TYPE_FUNC_NAME_PAIR, POD_DATA_TYPE_SEQ)
#undef DATA_TYPE_FUNC_NAME_PAIR
  };
  return JUST(MapAt(data_type2func_name, static_cast<int64_t>(dtype)));
}

Maybe<std::tuple<std::vector<Shape>, std::vector<Symbol<DType>>>>
MaybeGetTensorBufferShapesAndDTypes(const std::shared_ptr<Tensor>& t) {
  const auto& tensor = JUST(t->AsMirroredTensor());
  if (tensor->dtype() != DType::TensorBuffer()) {
    return Error::RuntimeError() << "tensor buffer supported only";
  }
  CHECK_OR_RETURN(tensor->is_eager()) << "eager tensors supported only";
  std::vector<Shape> shapes;
  std::vector<Symbol<DType>> dtypes;

  auto btb = std::make_shared<BlockingThenBusy>(1);
  JUST(PhysicalRun([&](InstructionsBuilder* builder) -> Maybe<void> {
    return builder->SyncAccessBlobByCallback(
        tensor, btb, [](uint64_t) {}, "const");
  }));
  JUST(btb->WaitUntilCntEqualZero(VirtualMachine::GetPredicatorNoMoreInstructionsFinished()));

  const Blob& blob = JUST(tensor->eager_blob_object())->blob();
  const Shape& blob_shape = blob.static_shape();
  const auto* tensor_buffer_ptr = blob.dptr<TensorBuffer>();
  for (int64_t i = 0; i < blob_shape.elem_cnt(); ++i) {
    const TensorBuffer* tensor_buffer = tensor_buffer_ptr + i;
    shapes.emplace_back(tensor_buffer->shape());
    dtypes.emplace_back(DType::Get(tensor_buffer->data_type()).GetOrThrow());
  }
  return std::make_tuple(shapes, dtypes);
}

Maybe<void> RegisterTensorHook(const std::shared_ptr<Tensor>& self,
                               const AutogradMeta::Hook& hook) {
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

#define MAKE_SWITCH_ENTRY(func_name, dtype) func_name<dtype>
DEFINE_STATIC_SWITCH_FUNC(Maybe<void>, CopyMirroredTensorFromUntypedArray, MAKE_SWITCH_ENTRY,
                          MAKE_DATA_TYPE_CTRV_SEQ(POD_AND_HALF_DATA_TYPE_SEQ));

Maybe<Tensor> MakeLocalTensorFromData(PyObject* data, const Optional<Symbol<DType>>& dtype,
                                      const Optional<Symbol<Device>>& device, bool requires_grad) {
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

  Symbol<Device> device_;
  if (device) {
    device_ = JUST(device);
  } else {
    device_ = JUST(Device::New("cpu"));
  }
  std::shared_ptr<Tensor> tensor =
      JUST(functional::Empty(shape, JUST(DType::Get(data_type)), device_));
  JUST(SwitchCopyMirroredTensorFromUntypedArray(SwitchCase(data_type), tensor, array));

  Py_DECREF(array);
  // Cast to float if data is double sequence, rather than numpy array.
  Symbol<DType> dtype_;
  if (dtype) {
    dtype_ = JUST(dtype);
  } else if (!dtype && data_type == DataType::kDouble && !PyArray_Check(data)) {
    dtype_ = DType::Float();
  }
  if (dtype_) { tensor = JUST(functional::Cast(tensor, dtype_)); }
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

Maybe<Tensor> MakeConsistentTensorFromData(PyObject* data, const Optional<Symbol<DType>>& dtype,
                                           Symbol<ParallelDesc> placement,
                                           const std::vector<Symbol<SbpParallel>>& sbp_tuple,
                                           bool requires_grad) {
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

  const std::string& device_tag = placement->device_tag();
  Symbol<Device> device;
  if (device_tag == "cpu") {
    device = JUST(Device::New("cpu"));
  } else {
    device = JUST(Device::New("cuda"));
  }
  std::shared_ptr<Tensor> local_tensor =
      JUST(functional::Empty(shape, JUST(DType::Get(data_type)), device));
  JUST(SwitchCopyMirroredTensorFromUntypedArray(SwitchCase(data_type), local_tensor, array));

  Py_DECREF(array);
  // Cast to float if data is double sequence, rather than numpy array.
  Symbol<DType> dtype_;
  if (dtype) {
    dtype_ = JUST(dtype);
  } else if (!dtype && data_type == DataType::kDouble && !PyArray_Check(data)) {
    dtype_ = DType::Float();
  }
  if (dtype_) { local_tensor = JUST(functional::Cast(local_tensor, dtype_)); }

  size_t sbp_dims = sbp_tuple.size();
  Symbol<NdSbp> broadcast_nd_sbp = JUST(CachedGetAllBroadcastNdSbp(sbp_dims));

  std::shared_ptr<Tensor> broadcast_tensor = JUST(functional::LocalToConsistent(
      local_tensor, placement, *JUST(GetSbpList(broadcast_nd_sbp)), shape, local_tensor->dtype()));

  std::vector<Symbol<SbpParallel>> grad_sbp_tuple;
  auto consistent_tensor =
      JUST(functional::ToConsistent(broadcast_tensor, placement, sbp_tuple, grad_sbp_tuple));
  JUST(consistent_tensor->set_requires_grad(requires_grad));
  return consistent_tensor;
}

Maybe<Tensor> MakeTensorFromOtherTensor(const std::shared_ptr<Tensor>& other) {
  if (other->is_local()) {
    const Symbol<Device>& device = JUST(other->device());
    return functional::Copy(other, device->type(), device->device_id());
  } else {
    const Symbol<NdSbp>& nd_sbp = JUST(other->nd_sbp());
    std::vector<Symbol<SbpParallel>> sbp_tuple(nd_sbp->sbp_parallel().size());
    for (int i = 0; i < sbp_tuple.size(); ++i) { sbp_tuple[i] = nd_sbp->sbp_parallel().Get(i); }
    std::vector<Symbol<SbpParallel>> grad_sbp_tuple;
    return functional::ToConsistent(other, JUST(other->parallel_desc()), sbp_tuple, grad_sbp_tuple);
  }
}

Maybe<Tensor> MakeTensorFromOtherTensor(const std::shared_ptr<Tensor>& other,
                                        const Optional<Symbol<DType>>& dtype,
                                        const Optional<Symbol<Device>>& device,
                                        const bool& requires_grad) {
  std::shared_ptr<Tensor> tensor;
  Symbol<Device> device_;
  if (device) { device_ = JUST(device); }
  if (other->is_local()) {
    if (!device) { device_ = JUST(other->device()); }
    tensor = JUST(functional::Copy(other, device_->type(), device_->device_id()));
  } else {
    tensor = JUST(functional::ConsistentToLocal(other));
    if (!device) { device_ = JUST(Device::New("cpu")); }
    tensor = JUST(functional::Copy(tensor, device_->type(), device_->device_id()));
  }
  if (dtype) {
    const Symbol<DType>& dtype_ = JUST(dtype);
    if (tensor->dtype() != dtype_) { tensor = JUST(functional::Cast(tensor, dtype_)); }
  }
  JUST(tensor->set_requires_grad(requires_grad));
  return tensor;
}

Maybe<Tensor> MakeTensorFromOtherTensor(const std::shared_ptr<Tensor>& other,
                                        const Optional<Symbol<DType>>& dtype,
                                        const Symbol<ParallelDesc>& placement,
                                        const std::vector<Symbol<SbpParallel>>& sbp_tuple,
                                        const bool& requires_grad) {
  std::vector<Symbol<SbpParallel>> grad_sbp_tuple;
  std::shared_ptr<Tensor> tensor =
      JUST(functional::ToConsistent(other, placement, sbp_tuple, grad_sbp_tuple));
  if (dtype) {
    const Symbol<DType>& dtype_ = JUST(dtype);
    if (tensor->dtype() != dtype_) { tensor = JUST(functional::Cast(tensor, dtype_)); }
  }
  JUST(tensor->set_requires_grad(requires_grad));
  return tensor;
}

}  // namespace one
}  // namespace oneflow
