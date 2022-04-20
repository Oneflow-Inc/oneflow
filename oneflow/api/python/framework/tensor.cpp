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
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>

#include "oneflow/core/common/throw.h"
#include "oneflow/api/python/framework/size.h"
#include "oneflow/api/python/of_api_registry.h"
#include "oneflow/api/python/ofblob/ofblob.e.h"
#include "oneflow/api/python/utils/tensor_utils.h"
#include "oneflow/api/python/functional/tensor_api.yaml.pybind.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/tensor_rpc_util.h"
#include "oneflow/core/framework/device.h"
#include "oneflow/core/framework/stride.h"
#include "oneflow/core/framework/py_distribute.h"
#include "oneflow/core/job/placement.cfg.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/framework/dtype.h"
#include "oneflow/core/framework/placement_utils.h"
#include "oneflow/core/autograd/autograd_engine.h"
#include "oneflow/core/common/decorator.h"

namespace py = pybind11;

namespace oneflow {

namespace one {

namespace {

const Symbol<DType>* GetTensorDType(const Tensor& tensor) {
  return &CHECK_JUST(DType::Get(tensor.dtype()->data_type()));
}

Maybe<py::array> ApiEagerMirroredTensorToNumpy(const py::handle& py_tensor) {
  const std::shared_ptr<Tensor> tensor = py::cast<const std::shared_ptr<Tensor>>(py_tensor);
  DataType data_type = tensor->dtype()->data_type();
  switch (data_type) {
#define SWITCH_EAGER_TENSOR_TO_NUMPY(cpp_type, of_type) \
  case of_type: return EagerMirroredTensorToNumpy<cpp_type>(py_tensor);
    OF_PP_FOR_EACH_TUPLE(SWITCH_EAGER_TENSOR_TO_NUMPY, POD_DATA_TYPE_SEQ)
    case DataType::kFloat16: return EagerMirroredTensorToNumpy<float16>(py_tensor);
    default: return Maybe<py::array>(Error::UnimplementedError() << "Invalid datatype");
  }
}

template<typename T>
Maybe<void> CopyMirroredTensorToNumpy(const std::shared_ptr<Tensor>& tensor, py::array_t<T> array) {
  return CopyBetweenMirroredTensorAndNumpy<T>(tensor->contiguous(), array.ptr(),
                                              BlobNumpyCopyUtil<T>::To, "const",
                                              /*block_host_until_done=*/true);
}

template<typename T>
Maybe<void> CopyMirroredTensorFromNumpy(const std::shared_ptr<Tensor>& tensor,
                                        py::array_t<T> array) {
  // When asynchronously copying array data to tensor, we need to back up the
  // array at the same time.
  // Only NPY_CORDER is supported, and it makes sure that the array is C-style contiguous.
  auto* copied_array = PyArray_NewCopy((PyArrayObject*)array.ptr(), NPY_CORDER);
  JUST(CopyBetweenMirroredTensorAndNumpy<T>(tensor, copied_array, BlobNumpyCopyUtil<T>::From, "mut",
                                            /*block_host_until_done=*/false));

  Py_DECREF(copied_array);
  return Maybe<void>::Ok();
}

std::shared_ptr<Tensor> ApiNewTensor(py::args args, py::kwargs kwargs) {
  return py::cast<std::shared_ptr<Tensor>>(functional::_legacy_tensor_ctor(args, kwargs));
}

Maybe<void> ApiSetRequiresGrad(Tensor& tensor, bool requires_grad) {
  CHECK_OR_RETURN(tensor.is_leaf())
      << Error::RuntimeError() << "You can only change requires_grad flags of leaf tensors.";
  JUST(tensor.set_requires_grad(requires_grad));
  return Maybe<void>::Ok();
}

std::shared_ptr<Parameter> ApiNewParameter(const std::shared_ptr<Tensor>& data,
                                           bool requires_grad) {
  return Parameter::MakeTensor(data, requires_grad).GetPtrOrThrow();
}

}  // namespace

using namespace pybind11::literals;

ONEFLOW_API_PYBIND11_MODULE("", m) {
  py::class_<Tensor, std::shared_ptr<Tensor>>(m, "Tensor")
      .def(py::init(&ApiNewTensor))
      // Properties of pytorch
      .def_property_readonly("ndim", &Tensor::ndim)
      .def_property_readonly("shape", &Tensor::shape)
      .def_property_readonly("dtype", &GetTensorDType)
      .def_property_readonly("is_cuda", &Tensor::is_cuda)
      .def_property("grad", &Tensor::acc_grad,
                    [](Tensor& t, const std::shared_ptr<Tensor>& grad) -> Maybe<void> {
                      CHECK_OR_RETURN(t.is_leaf())
                          << Error::RuntimeError()
                          << "You can only change gradient of leaf tensors.";
                      if (grad != nullptr) {
                        JUST(t.set_acc_grad(JUST(grad->detach())));
                      } else {
                        JUST(t.set_acc_grad(nullptr));
                      }
                      return Maybe<void>::Ok();
                    })
      .def_property(
          "_is_grad_acc_inplace",
          [](const Tensor& t) -> bool { return t.autograd_meta()->is_grad_acc_inplace(); },
          [](Tensor& t, bool is_inplace) {
            t.mut_autograd_meta()->set_is_grad_acc_inplace(is_inplace);
          })
      .def_property("data", &Tensor::data,
                    [](const std::shared_ptr<Tensor>& t,
                       const std::shared_ptr<Tensor>& other) -> Maybe<void> {
                      auto hooks = t->autograd_meta()->hooks();
                      JUST(t->set_data(other));
                      // Re-register hooks
                      for (const auto& hook : hooks) { JUST(RegisterTensorHook(t, hook)); }
                      return Maybe<void>::Ok();
                    })
      .def("storage_offset", &Tensor::storage_offset)
      .def("stride",
           [](const Tensor& t) -> Maybe<py::tuple> {
             const auto& stride = JUST(t.stride())->StrideVec();
             return py::tuple(py::make_iterator(stride.begin(), stride.end()));
           })
      .def("is_contiguous", &Tensor::is_contiguous)
      .def("contiguous", &Tensor::contiguous)
      .def_property_readonly("grad_fn", &Tensor::grad_fn_node)
      .def_property_readonly("is_leaf", &Tensor::is_leaf)
      .def_property("requires_grad", &Tensor::requires_grad, &ApiSetRequiresGrad)
      // Methods of pytorch
      .def(
          "requires_grad_",
          [](Tensor& t, bool requires_grad) -> Maybe<Tensor&> {
            JUST(ApiSetRequiresGrad(t, requires_grad));
            return t;
          },
          "requires_grad"_a = true)
      .def("retain_grad",
           [](Tensor& t) -> Maybe<void> {
             if (!t.is_leaf()) { JUST(t.set_retain_grad(true)); }
             return Maybe<void>::Ok();
           })
      .def("detach", &Tensor::detach)
      .def("clone", &Tensor::clone)
      // OneFlow tensor properties other than pytorch tensor
      .def_property_readonly("is_lazy", &Tensor::is_lazy)
      .def_property_readonly("is_eager", &Tensor::is_eager)
      .def_property_readonly("is_global", &Tensor::is_consistent)
      .def_property_readonly("is_local", &Tensor::is_local)
      .def("zeros_", &EagerMirroredTensorZeros)
      .def("register_hook", &RegisterTensorHook)
      .def("_register_post_grad_accumulation_hook", &RegisterTensorPostGradAccumulationHook)
      // local tensor only
      .def_property_readonly("_tensor_buffer_shapes_and_dtypes",
                             &MaybeGetTensorBufferShapesAndDTypes)
      .def_property_readonly("device", &Tensor::device)
      .def("global_id",
           [](const one::Tensor& tensor) -> Maybe<int64_t> {
             return static_cast<uint64_t>(JUST(tensor.transport_token()));
           })
      .def("check_meta_consistency", CheckMetaConsistency)
      .def("to_numpy", &ApiEagerMirroredTensorToNumpy, py::return_value_policy::move)
#define DEFINE_TENSOR_METHOD(T, type_proto)                 \
  .def("_copy_to_numpy_" #T, &CopyMirroredTensorToNumpy<T>) \
      .def("_copy_from_numpy_" #T, &CopyMirroredTensorFromNumpy<T>)
          OF_PP_FOR_EACH_TUPLE(DEFINE_TENSOR_METHOD, POD_DATA_TYPE_SEQ)
#undef DEFINE_TENSOR_METHOD
      .def("_get_copy_mirrored_tensor_to_numpy_func_name",
           [](const Tensor& tensor) {
             return GetCopyMirroredTensorToNumpyFuncName(tensor.dtype()->data_type());
           })
      .def("_get_copy_mirrored_tensor_from_numpy_func_name",
           [](const Tensor& tensor) {
             return GetCopyMirroredTensorFromNumpyFuncName(tensor.dtype()->data_type());
           })
      .def("_register_storage_delete_hook", &Tensor::RegisterStorageDeleteHook)
      // consistent tensor only
      .def_property_readonly("placement", &Tensor::parallel_desc)
      .def_property_readonly("sbp", &TensorGetPyTupleOfSbp);

  auto nn = m.def_submodule("nn");
  py::class_<Parameter, std::shared_ptr<Parameter>, Tensor>(nn, "Parameter")
      .def(py::init(&ApiNewParameter), "data"_a, "requires_grad"_a = true);
}

}  // namespace one

}  // namespace oneflow
