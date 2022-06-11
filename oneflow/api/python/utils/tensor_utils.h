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
#ifndef ONEFLOW_API_PYTHON_UTILS_TENSOR_UTILS_H_
#define ONEFLOW_API_PYTHON_UTILS_TENSOR_UTILS_H_

#include <Python.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>

#include "oneflow/api/python/framework/tensor.h"
#include "oneflow/extension/python/numpy.h"
#include "oneflow/core/framework/device.h"
#include "oneflow/core/framework/dtype.h"
#include "oneflow/core/framework/instructions_builder.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/common/stride.h"
#include "oneflow/core/register/ofblob.h"
#include "oneflow/core/common/blocking_then_busy.h"
#include "oneflow/core/vm/virtual_machine.h"
#include "oneflow/core/common/foreign_lock_helper.h"

namespace py = pybind11;

namespace pybind11 {
// reference: https://github.com/pybind/pybind11/issues/1776
template<>
struct format_descriptor<oneflow::float16> {
  static pybind11::dtype dtype() {
    handle ptr = detail::npy_api::get().PyArray_DescrFromType_(NPY_FLOAT16);
    return reinterpret_borrow<pybind11::dtype>(ptr);
  }
  static std::string format() {
    // following: https://docs.python.org/3/library/struct.html#format-characters
    return "e";
  }
  static constexpr auto name() { return detail::_("float16"); }
};
}  // namespace pybind11

namespace oneflow {
namespace one {

Maybe<void> EagerMirroredTensorZeros(const std::shared_ptr<Tensor>& t);

template<typename T>
inline static Maybe<PyObject*> EagerMirroredTensorToNumpy(PyObject* py_tensor) {
  const auto& t = PyTensor_Unpack(py_tensor);

  std::shared_ptr<MirroredTensor> tensor = JUST(t->AsMirroredTensor());
  CHECK_OR_RETURN(JUST(tensor->device()) == JUST(Device::New("cpu")));
  CHECK_OR_RETURN(tensor->is_eager()) << "eager tensors supported only.";
  // set base object attr
  py::handle handle = py::handle(py_tensor);

  const size_t ndim = tensor->ndim();
  const auto shape = numpy::OFShapeToNumpyShape(tensor->shape()->dim_vec());
  // NumPy strides use bytes. OneFlow strides use element counts.
  const auto stride =
      numpy::OFStrideToNumpyStride(*JUST(tensor->stride()), tensor->dtype()->data_type());

  T* data_ptr = nullptr;
  const auto& Callback = [&](uint64_t ofblob_ptr) {
    data_ptr = reinterpret_cast<OfBlob*>(ofblob_ptr)->mut_blob()->mut_dptr<T>();
  };
  auto btb = std::make_shared<BlockingThenBusy>(1);
  JUST(PhysicalRun([&](InstructionsBuilder* builder) -> Maybe<void> {
    return builder->SyncAccessBlobByCallback(tensor, btb, Callback, "mut");
  }));
  JUST(btb->WaitUntilCntEqualZero(VirtualMachine::GetPredicatorNoMoreInstructionsFinished()));
  return py::array(py::buffer_info(data_ptr, sizeof(T), py::format_descriptor<T>::format(), ndim,
                                   shape, stride),
                   handle)
      .release()
      .ptr();
}

template<typename T>
inline Maybe<void> CopyBetweenMirroredTensorAndNumpy(
    const std::shared_ptr<Tensor>& t, PyObject* array,
    Maybe<void> (*Copy)(uint64_t, const NumPyArrayPtr&), const std::string& modifier,
    bool block_host_until_done) {
  auto tensor = JUST(t->AsMirroredTensor());
  CHECK_OR_RETURN(tensor->is_eager()) << "eager tensors supported only.";

  if (block_host_until_done) {
    NumPyArrayPtr array_ptr(array);
    const auto& Callback = [array_ptr, Copy](uint64_t ofblob_ptr) {
      CHECK_JUST(Copy(ofblob_ptr, array_ptr));
    };
    auto btb = std::make_shared<BlockingThenBusy>(1);
    JUST(PhysicalRun([&](InstructionsBuilder* builder) -> Maybe<void> {
      return builder->SyncAccessBlobByCallback(tensor, btb, Callback, modifier);
    }));
    JUST(btb->WaitUntilCntEqualZero(VirtualMachine::GetPredicatorNoMoreInstructionsFinished()));
  } else {
    Py_INCREF(array);
    NumPyArrayPtr array_ptr(array, [array]() {
      CHECK_JUST(Global<ForeignLockHelper>::Get()->WithScopedAcquire([&]() -> Maybe<void> {
        Py_DECREF(array);
        return Maybe<void>::Ok();
      }));
    });

    JUST(PhysicalRun([&](InstructionsBuilder* builder) -> Maybe<void> {
      return builder->AccessBlobByCallback(
          tensor,
          [array_ptr, Copy](uint64_t ofblob_ptr) { CHECK_JUST(Copy(ofblob_ptr, array_ptr)); },
          modifier);
    }));
  }
  return Maybe<void>::Ok();
}

Maybe<std::string> GetCopyMirroredTensorToNumpyFuncName(DataType dtype);

Maybe<std::string> GetCopyMirroredTensorFromNumpyFuncName(DataType dtype);

Maybe<std::tuple<std::vector<Shape>, std::vector<Symbol<DType>>>>
MaybeGetTensorBufferShapesAndDTypes(const std::shared_ptr<Tensor>& t);

Maybe<void> RegisterTensorHook(const std::shared_ptr<Tensor>& self, const AutogradMeta::Hook& hook);

Maybe<void> RegisterTensorPostGradAccumulationHook(const std::shared_ptr<Tensor>& self,
                                                   const AutogradMeta::Hook& hook);

Maybe<py::tuple> TensorGetPyTupleOfSbp(const Tensor& tensor);

Maybe<Tensor> MakeLocalTensorFromData(PyObject* data, const Optional<Symbol<DType>>& dtype,
                                      const Optional<Symbol<Device>>& device,
                                      const bool requires_grad, const bool pin_memory);

Maybe<Tensor> MakeConsistentTensorFromData(PyObject* data, const Optional<Symbol<DType>>& dtype,
                                           Symbol<ParallelDesc> placement,
                                           const std::vector<Symbol<SbpParallel>>& sbp_tuple,
                                           const bool requires_grad);

Maybe<Tensor> MakeTensorFromOtherTensor(const std::shared_ptr<Tensor>& other,
                                        const bool pin_memory);

Maybe<Tensor> MakeTensorFromOtherTensor(const std::shared_ptr<Tensor>& other,
                                        const Optional<Symbol<DType>>& dtype,
                                        const Optional<Symbol<Device>>& device,
                                        const bool requires_grad, const bool pin_memory);

Maybe<Tensor> MakeTensorFromOtherTensor(const std::shared_ptr<Tensor>& other,
                                        const Optional<Symbol<DType>>& dtype,
                                        const Symbol<ParallelDesc>& placement,
                                        const std::vector<Symbol<SbpParallel>>& sbp_tuple,
                                        const bool requires_grad);

}  // namespace one
}  // namespace oneflow

#endif  // ONEFLOW_API_PYTHON_UTILS_TENSOR_UTILS_H_
