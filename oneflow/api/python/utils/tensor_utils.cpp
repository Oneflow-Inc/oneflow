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

namespace py = pybind11;

namespace oneflow {
namespace one {

Maybe<void> EagerMirroredTensorZeros(const std::shared_ptr<Tensor>& t) {
  const auto& tensor = JUST(t->AsMirroredTensor());
  CHECK_OR_RETURN(tensor->is_eager()) << "eager tensors supported only";
  JUST(PhysicalRun([&](InstructionsBuilder* builder) -> Maybe<void> {
    JUST(builder->AccessBlobByCallback(
        tensor,
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
                                               py::object array) {
  return CopyBetweenMirroredTensorAndNumpy(tensor, array.cast<py::array_t<T>>(),
                                           OfBlob_CopyFromBuffer, "mut");
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
  CHECK_OR_RETURN(tensor->is_eager()) << "eager tensors supported only";
  std::vector<Shape> shapes;
  std::vector<Symbol<DType>> dtypes;

  const auto& Callback = std::make_shared<std::function<void(uint64_t)>>([](uint64_t) {});
  JUST(SpinCounter::SpinWait(1, [&](const std::shared_ptr<SpinCounter>& sc) -> Maybe<void> {
    return PhysicalRun([&](InstructionsBuilder* builder) -> Maybe<void> {
      return builder->SyncAccessBlobByCallback(tensor, sc, Callback, "const");
    });
  }));

  const Blob& blob = JUST(tensor->eager_blob_object())->blob();
  const Shape& blob_shape = blob.static_shape();
  const auto* tensor_buffer_ptr = blob.dptr<TensorBuffer>();
  for (int64_t i = 0; i < blob_shape.elem_cnt(); ++i) {
    const TensorBuffer* tensor_buffer = tensor_buffer_ptr + i;
    shapes.push_back(tensor_buffer->shape());
    dtypes.push_back(DType::Get(tensor_buffer->data_type()).GetOrThrow());
  }
  return std::make_tuple(shapes, dtypes);
}

Maybe<void> RegisterTensorHook(const std::shared_ptr<Tensor>& self,
                               const AutogradMeta::Hook& hook) {
  if (!self->grad_fn_node()) { JUST(AddAccumulateFunctionNode(self)); }
  self->mut_autograd_meta()->add_hook(hook);
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
                          MAKE_DATA_TYPE_CTRV_SEQ(POD_DATA_TYPE_SEQ));

Maybe<Tensor> MakeLocalTensorFromData(PyObject* data, const Optional<Symbol<DType>>& dtype,
                                      const Optional<Symbol<Device>>& device, bool requires_grad) {
  // Executing any numpy c api before _import_array() results in segfault
  if (PyArray_API == nullptr) { _import_array(); }
  auto* np_arr_pyobject = PyArray_FromAny(data, nullptr, 0, 0, NPY_ARRAY_DEFAULT, nullptr);
  if (!np_arr_pyobject) {
    return Error::RuntimeError() << "Can not convert input data to a numpy array.";
  }
  // transfer the ownership to np_arr_raii so that the ref count
  // can be decreased automatically when function exits either normally or abnormally
  auto np_arr_raii = py::reinterpret_steal<py::array>(np_arr_pyobject);
  auto* np_arr = reinterpret_cast<PyArrayObject*>(np_arr_pyobject);
  const npy_intp* dims_ptr = PyArray_SHAPE(np_arr);
  const Shape shape(DimVector(dims_ptr, dims_ptr + PyArray_NDIM(np_arr)));
  DataType data_type = JUST(numpy::GetOFDataTypeFromNpArray(np_arr));

  Symbol<Device> device_;
  if (device) {
    device_ = JUST(device.value());
  } else {
    device_ = JUST(Device::New("cpu"));
  }
  std::shared_ptr<Tensor> tensor =
      JUST(functional::Empty(shape, CHECK_JUST(DType::Get(data_type)), device_));
  JUST(SwitchCopyMirroredTensorFromUntypedArray(SwitchCase(data_type), tensor, np_arr_raii));

  // Cast to float if data is double sequence, rather than numpy array.
  Symbol<DType> dtype_;
  if (dtype) {
    dtype_ = JUST(dtype.value());
  } else if (!dtype && data_type == DataType::kDouble && !PyArray_Check(data)) {
    dtype_ = DType::Float();
  }
  if (dtype_) { tensor = JUST(functional::Cast(tensor, dtype_)); }
  tensor->set_requires_grad(requires_grad);
  return tensor;
}

Maybe<Tensor> MakeTensorFromOtherTensor(const std::shared_ptr<Tensor>& other) {
  if (other->is_local()) {
    const Symbol<Device>& device = JUST(other->device());
    return functional::Copy(other, device->type(), device->device_id());
  } else {
    const Symbol<cfg::NdSbp>& nd_sbp = JUST(other->nd_sbp());
    std::vector<Symbol<cfg::SbpParallel>> sbp_tuple(nd_sbp->sbp_parallel().size());
    for (int i = 0; i < sbp_tuple.size(); ++i) { sbp_tuple[i] = nd_sbp->sbp_parallel().Get(i); }
    return functional::ToConsistent(other, JUST(other->parallel_desc()), sbp_tuple,
                                    GetNoneSbpList());
  }
}

Maybe<Tensor> MakeTensorFromOtherTensor(
    const std::shared_ptr<Tensor>& other, const Optional<Symbol<DType>>& dtype,
    const Optional<Symbol<Device>>& device, const Optional<Symbol<ParallelDesc>>& placement,
    const Optional<std::vector<Symbol<cfg::SbpParallel>>>& sbp_tuple, const bool& requires_grad) {
  if (placement) {
    if (!sbp_tuple) {
      return Error::RuntimeError() << "Sbp tuple is expected while constructing consistent tensor.";
    }
  }
  std::shared_ptr<Tensor> tensor;
  if (placement) {
    tensor = JUST(functional::ToConsistent(other, JUST(placement.value()), *JUST(sbp_tuple.value()),
                                           GetNoneSbpList()));
  } else {
    Symbol<Device> device_;
    if (device) { device_ = JUST(device.value()); }
    if (other->is_local()) {
      if (!device) { device_ = JUST(other->device()); }
      tensor = JUST(functional::Copy(other, device_->type(), device_->device_id()));
    } else {
      tensor = JUST(functional::ConsistentToLocal(other));
      if (!device) { device_ = JUST(Device::New("cpu")); }
      tensor = JUST(functional::Copy(tensor, device_->type(), device_->device_id()));
    }
  }
  if (dtype) {
    const Symbol<DType>& dtype_ = JUST(dtype.value());
    if (tensor->dtype() != dtype_) { return functional::Cast(tensor, dtype_); }
  }
  return tensor;
}

}  // namespace one
}  // namespace oneflow
