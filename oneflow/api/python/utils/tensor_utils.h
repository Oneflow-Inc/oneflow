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
#include "oneflow/core/common/blocking_then_busy.h"
#include "oneflow/core/vm/virtual_machine.h"
#include "oneflow/core/common/foreign_lock_helper.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/api/python/functional/common.h"
#include "oneflow/core/framework/tensor_util.h"
#include "oneflow/core/profiler/profiler.h"

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

Maybe<void> EagerLocalTensorZeros(const std::shared_ptr<Tensor>& t);

template<typename T>
inline static Maybe<PyObject*> EagerLocalTensorToNumpy(PyObject* py_tensor) {
  const auto& t = PyTensor_Unpack(py_tensor);

  std::shared_ptr<LocalTensor> tensor = JUST(t->AsLocalTensor());
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
  const auto& Callback = [&](ep::Stream*,
                             const std::shared_ptr<vm::EagerBlobObject>& eager_blob_object) {
    data_ptr = eager_blob_object->mut_dptr<T>();
  };
  auto btb = std::make_shared<BlockingThenBusy>();
  JUST(PhysicalRun([&](InstructionsBuilder* builder) -> Maybe<void> {
    return builder->SyncAccessBlobByCallback(tensor, btb, Callback, "const");
  }));
  JUST(btb->WaitUntilCntEqualZero(VirtualMachine::GetPredicatorNoMoreInstructionsFinished()));
  return py::array(py::buffer_info(data_ptr, sizeof(T), py::format_descriptor<T>::format(), ndim,
                                   shape, stride),
                   handle)
      .release()
      .ptr();
}

template<typename T>
struct TensorTypeToPyType final {
  typedef T type;
};

template<>
struct TensorTypeToPyType<float16> final {
  typedef float type;
};

template<>
struct TensorTypeToPyType<bfloat16> final {
  typedef float type;
};

template<typename T>
inline static Maybe<PyObject*> EagerLocalTensorItem(const std::shared_ptr<Tensor>& tensor) {
  // OF_PROFILER_RANGE_GUARD("EagerLocalTensorItem");
  T value = JUST(GetItemInScalarTensor<T>(tensor));
  return functional::CastToPyObject(static_cast<typename TensorTypeToPyType<T>::type>(value));
}

inline Maybe<void> CopyBetweenLocalTensorAndNumpy(
    const std::shared_ptr<Tensor>& t, PyObject* array,
    void (*Copy)(ep::Stream*, const std::shared_ptr<vm::EagerBlobObject>&, const NumPyArrayPtr&),
    const std::string& modifier, bool block_host_until_done) {
  auto tensor = JUST(t->AsLocalTensor());
  CHECK_OR_RETURN(tensor->is_contiguous()) << "contiguous tensors supported only.";
  CHECK_OR_RETURN(tensor->is_eager()) << "eager tensors supported only.";

  if (block_host_until_done) {
    NumPyArrayPtr array_ptr(array);
    const auto& Callback = [array_ptr, Copy](
                               ep::Stream* stream,
                               const std::shared_ptr<vm::EagerBlobObject>& eager_blob_object) {
      Copy(stream, eager_blob_object, array_ptr);
    };
    auto btb = std::make_shared<BlockingThenBusy>();
    JUST(PhysicalRun([&](InstructionsBuilder* builder) -> Maybe<void> {
      return builder->SyncAccessBlobByCallback(tensor, btb, Callback, modifier);
    }));
    JUST(btb->WaitUntilCntEqualZero(VirtualMachine::GetPredicatorNoMoreInstructionsFinished()));
  } else {
    Py_INCREF(array);
    NumPyArrayPtr array_ptr(array, [array]() {
      // release array in main thread to eliminate the time-consuming gil request
      CHECK_JUST(SingletonMaybe<VirtualMachine>())->add_main_thread_pending_task([array]() {
        Py_DECREF(array);
      });
    });

    JUST(PhysicalRun([&](InstructionsBuilder* builder) -> Maybe<void> {
      return builder->AccessBlobByCallback(
          tensor,
          [array_ptr, Copy](ep::Stream* stream,
                            const std::shared_ptr<vm::EagerBlobObject>& eager_blob_object) {
            Copy(stream, eager_blob_object, array_ptr);
          },
          modifier);
    }));
  }
  return Maybe<void>::Ok();
}

Maybe<std::tuple<std::vector<Shape>, std::vector<Symbol<DType>>>>
MaybeGetTensorBufferShapesAndDTypes(const std::shared_ptr<Tensor>& t);

Maybe<void> RegisterTensorHook(const std::shared_ptr<Tensor>& self, const AutogradMeta::Hook& hook);

Maybe<void> RegisterTensorPostGradAccumulationHook(const std::shared_ptr<Tensor>& self,
                                                   const AutogradMeta::Hook& hook);

Maybe<py::tuple> TensorGetPyTupleOfSbp(const Tensor& tensor);

Maybe<Tensor> MakeLocalTensorFromData(PyObject* data, const Optional<Symbol<DType>>& dtype,
                                      const Optional<Symbol<Device>>& device,
                                      const bool requires_grad, const bool pin_memory);

Maybe<Tensor> MakeGlobalTensorFromData(PyObject* data, const Optional<Symbol<DType>>& dtype,
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
