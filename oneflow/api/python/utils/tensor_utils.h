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

#include "oneflow/extension/python/numpy.h"
#include "oneflow/core/common/thread_local_callback.h"
#include "oneflow/core/framework/device.h"
#include "oneflow/core/framework/dtype.h"
#include "oneflow/core/framework/instructions_builder.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/nd_sbp.h"
#include "oneflow/core/functional/functional_api.yaml.h"
#include "oneflow/core/framework/stride.h"
#include "oneflow/core/register/ofblob.h"
#include "oneflow/extension/python/numpy.h"
namespace py = pybind11;

namespace oneflow {
namespace one {

Maybe<void> EagerMirroredTensorZeros(const std::shared_ptr<Tensor>& t);

template<typename T>
inline static Maybe<py::array> EagerTensorToNumpy(const py::handle& py_tensor) {
  const std::shared_ptr<Tensor> t = py::cast<const std::shared_ptr<Tensor>>(py_tensor);

  std::shared_ptr<MirroredTensor> tensor = JUST(t->AsMirroredTensor());
  CHECK_OR_RETURN(JUST(tensor->device()) == JUST(Device::New("cpu")));
  CHECK_OR_RETURN(tensor->is_eager()) << "eager tensors supported only";
  // set base object attr
  py::handle handle = py::handle(py_tensor.ptr());

  const size_t ndim = tensor->ndim();
  const auto shape = numpy::OFShapeToNumpyShape(tensor->shape()->dim_vec());
  // NumPy strides use bytes. OneFlow strides use element counts.
  const auto stride = numpy::OFStrideToNumpyStride(JUST(tensor->stride())->StrideVec(),
                                                   tensor->dtype()->data_type());

  T* data_ptr = nullptr;
  const auto& Callback = std::make_shared<std::function<void(uint64_t)>>([&](uint64_t ofblob_ptr) {
    data_ptr = reinterpret_cast<OfBlob*>(ofblob_ptr)->mut_blob()->mut_dptr<T>();
  });
  bool is_printed = false;
  SpinCounter::SpinWait(
      1,
      [&](const std::shared_ptr<SpinCounter>& sc) -> Maybe<void> {
        return PhysicalRun([&](InstructionsBuilder* builder) -> Maybe<void> {
          return builder->SyncAccessBlobByCallback(tensor, sc, Callback, "mut");
        });
      },
      [&is_printed]() {
        if (!is_printed) {
          blocking::StackInfoCallback();
          is_printed = true;
        }
      });

  return py::array(
      py::buffer_info(data_ptr, sizeof(T), py::format_descriptor<T>::format(), ndim, shape, stride),
      handle);
}

template<typename T>
inline Maybe<void> CopyBetweenMirroredTensorAndNumpy(
    const std::shared_ptr<Tensor>& t, PyObject* array,
    Maybe<void> (*Copy)(uint64_t, const NumPyArrayPtr&), const std::string& modifier,
    bool block_host_until_done) {
  std::shared_ptr<MirroredTensor> tensor;
  CHECK_OR_RETURN(t->is_eager()) << "eager tensors supported only";
  if (t->is_local()) {
    tensor = JUST(t->AsMirroredTensor());
  } else {
    const Symbol<ConsistentTensorMeta>& tensor_meta = JUST(t->consistent_tensor_meta());
    const Symbol<cfg::NdSbp>& nd_sbp = tensor_meta->nd_sbp();
    CHECK_OR_RETURN(!nd_sbp->sbp_parallel().empty());
    cfg::SbpParallel broadcast_sbp;
    broadcast_sbp.mutable_broadcast_parallel();
    std::vector<Symbol<cfg::SbpParallel>> sbp_tuple(nd_sbp->sbp_parallel_size(),
                                                    SymbolOf(broadcast_sbp));
    std::vector<Symbol<cfg::SbpParallel>> none;
    const auto& consistent_tensor =
        JUST(functional::ToConsistent(t, tensor_meta->parallel_desc(), sbp_tuple, none));
    tensor = JUST(consistent_tensor->cur_rank_phy_tensor());
  }

  if (block_host_until_done) {
    NumPyArrayPtr array_ptr(array);
    const auto& Callback = std::make_shared<std::function<void(uint64_t)>>(
        [array_ptr, Copy](uint64_t ofblob_ptr) { CHECK_JUST(Copy(ofblob_ptr, array_ptr)); });
    bool is_printed = false;
    JUST(SpinCounter::SpinWait(
        1,
        [&](const std::shared_ptr<SpinCounter>& sc) -> Maybe<void> {
          return PhysicalRun([&](InstructionsBuilder* builder) -> Maybe<void> {
            return builder->SyncAccessBlobByCallback(tensor, sc, Callback, modifier);
          });
        },
        [&is_printed]() {
          if (!is_printed) {
            blocking::StackInfoCallback();
            is_printed = true;
          }
        }));
  } else {
    Py_INCREF(array);
    NumPyArrayPtr array_ptr(array, [array]() {
      py::gil_scoped_acquire acquire;
      Py_DECREF(array);
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

Maybe<py::tuple> TensorGetPyTupleOfSbp(const Tensor& tensor);

Maybe<Tensor> MakeLocalTensorFromData(PyObject* data, const Optional<Symbol<DType>>& dtype,
                                      const Optional<Symbol<Device>>& device, bool requires_grad);

Maybe<Tensor> MakeConsistentTensorFromData(PyObject* data, const Optional<Symbol<DType>>& dtype,
                                           Symbol<ParallelDesc> placement,
                                           const std::vector<Symbol<cfg::SbpParallel>>& sbp_tuple,
                                           bool requires_grad);

Maybe<Tensor> MakeTensorFromOtherTensor(const std::shared_ptr<Tensor>& other);

Maybe<Tensor> MakeTensorFromOtherTensor(const std::shared_ptr<Tensor>& other,
                                        const Optional<Symbol<DType>>& dtype,
                                        const Optional<Symbol<Device>>& device,
                                        const bool& requires_grad);

Maybe<Tensor> MakeTensorFromOtherTensor(const std::shared_ptr<Tensor>& other,
                                        const Optional<Symbol<DType>>& dtype,
                                        const Symbol<ParallelDesc>& placement,
                                        const std::vector<Symbol<cfg::SbpParallel>>& sbp_tuple,
                                        const bool& requires_grad);

}  // namespace one
}  // namespace oneflow

#endif  // ONEFLOW_API_PYTHON_UTILS_TENSOR_UTILS_H_
