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
#include "oneflow/api/python/of_api_registry.h"
#include "oneflow/api/python/ofblob/ofblob.e.h"
#include "oneflow/core/common/container_util.h"
#include "oneflow/core/common/tensor_buffer.h"
#include "oneflow/core/framework/instructions_builder.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/device.h"
#include "oneflow/core/framework/py_distribute.h"
#include "oneflow/core/job/placement.cfg.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/framework/dtype.h"
#include "oneflow/core/autograd/autograd_engine.h"
#include "oneflow/core/autograd/autograd_meta.h"

namespace py = pybind11;

namespace oneflow {

namespace one {

namespace {

const DType* GetTensorDType(const Tensor& tensor) {
  return DType::Get(tensor.dtype()).GetOrThrow().get();
}

std::shared_ptr<Tensor> MakeLocalTensor(const std::shared_ptr<const Shape>& shape,
                                        const DType* dtype, const Symbol<Device>& device,
                                        bool is_lazy, bool requires_grad, bool is_leaf) {
  return MirroredTensor::MakeTensor(shape, dtype->data_type(), device, is_lazy, requires_grad,
                                    is_leaf)
      .GetPtrOrThrow();
}

std::shared_ptr<Tensor> MakeConsistentTensor(
    const std::shared_ptr<const Shape>& shape, const DType* dtype,
    Symbol<cfg::ParallelDistribution>& parallel_distribution, Symbol<ParallelDesc> parallel_desc,
    bool is_lazy, bool requires_grad, bool is_leaf) {
  return ConsistentTensor::MakeTensor(shape, dtype->data_type(), parallel_distribution,
                                      parallel_desc, is_lazy, requires_grad, is_leaf)
      .GetPtrOrThrow();
}

Maybe<void> EagerMirroredTensorZeros(const std::shared_ptr<Tensor>& t) {
  const auto& tensor = std::dynamic_pointer_cast<MirroredTensor>(t);
  CHECK_NOTNULL_OR_RETURN(tensor) << "local tensors supported only";
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

void ApiEagerMirroredTensorZeros(const std::shared_ptr<Tensor>& tensor) {
  return EagerMirroredTensorZeros(tensor).GetOrThrow();
}

template<typename T>
Maybe<void> CopyBetweenMirroredTensorAndNumpy(const std::shared_ptr<Tensor>& t,
                                              py::array_t<T> array,
                                              void (*Copy)(uint64_t, py::array_t<T>),
                                              const std::string& modifier) {
  const auto& tensor = std::dynamic_pointer_cast<MirroredTensor>(t);
  CHECK_NOTNULL_OR_RETURN(tensor) << "local tensors supported only";
  CHECK_OR_RETURN(tensor->is_eager()) << "eager tensors supported only";
  std::atomic<bool> synced(false);

  JUST(PhysicalRun([&](InstructionsBuilder* builder) -> Maybe<void> {
    JUST(builder->AccessBlobByCallback(
        tensor,
        [&array, &synced, &Copy](uint64_t ofblob_ptr) {
          Copy(ofblob_ptr, array);
          synced = true;
        },
        modifier));
    return Maybe<void>::Ok();
  }));

  Global<ForeignLockHelper>::Get()->WithScopedRelease([&synced]() { /* spin wait */
                                                                    while (!synced) {}
  });

  return Maybe<void>::Ok();
}

template<typename T>
void ApiCopyMirroredTensorToNumpy(const std::shared_ptr<Tensor>& tensor, py::array_t<T> array) {
  return CopyBetweenMirroredTensorAndNumpy(tensor, array, OfBlob_CopyToBuffer, "const")
      .GetOrThrow();
}

template<typename T>
void ApiCopyMirroredTensorFromNumpy(const std::shared_ptr<Tensor>& tensor, py::array_t<T> array) {
  return CopyBetweenMirroredTensorAndNumpy(tensor, array, OfBlob_CopyFromBuffer, "mut")
      .GetOrThrow();
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

const std::string& ApiGetCopyMirroredTensorToNumpyFuncName(const Tensor& tensor) {
  return *GetCopyMirroredTensorToNumpyFuncName(tensor.dtype()).GetPtrOrThrow();
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

const std::string& ApiGetCopyMirroredTensorFromNumpyFuncName(const Tensor& tensor) {
  return *GetCopyMirroredTensorFromNumpyFuncName(tensor.dtype()).GetPtrOrThrow();
}

Symbol<Device> TensorGetDevice(const Tensor& tensor) {
  return tensor.device().GetOrThrow();
}

Symbol<ParallelDesc> TensorGetParallelDesc(const Tensor& tensor) {
  return tensor.parallel_desc().GetOrThrow();
}

Maybe<std::tuple<std::vector<Shape>, std::vector<const DType*>>>
MaybeGetTensorBufferShapesAndDTypes(const std::shared_ptr<Tensor>& t) {
  const auto& tensor = std::dynamic_pointer_cast<MirroredTensor>(t);
  CHECK_NOTNULL_OR_RETURN(tensor) << "local tensors supported only";
  CHECK_OR_RETURN(tensor->is_eager()) << "eager tensors supported only";
  std::vector<Shape> shapes;
  std::vector<const DType*> dtypes;
  std::atomic<bool> synced(false);

  JUST(PhysicalRun([&](InstructionsBuilder* builder) -> Maybe<void> {
    JUST(builder->AccessBlobByCallback(
        tensor, [&synced](uint64_t of_blob_ptr) { synced = true; }, "const"));
    return Maybe<void>::Ok();
  }));

  Global<ForeignLockHelper>::Get()->WithScopedRelease([&synced]() {
    while (!synced) {}
  });

  const Blob& blob = JUST(tensor->eager_blob_object())->blob();
  const Shape& blob_shape = blob.static_shape();
  const auto* tensor_buffer_ptr = blob.dptr<TensorBuffer>();
  for (int64_t i = 0; i < blob_shape.elem_cnt(); ++i) {
    const TensorBuffer* tensor_buffer = tensor_buffer_ptr + i;
    shapes.push_back(tensor_buffer->shape());
    dtypes.push_back(DType::Get(tensor_buffer->data_type()).GetOrThrow().get());
  }
  return std::make_tuple(shapes, dtypes);
}

std::tuple<std::vector<Shape>, std::vector<const DType*>> GetTensorBufferShapesAndDTypes(
    const std::shared_ptr<Tensor>& tensor) {
  return MaybeGetTensorBufferShapesAndDTypes(tensor).GetOrThrow();
}

Maybe<void> RegisterTensorHook(const std::shared_ptr<Tensor>& self,
                               const AutogradMeta::Hook& hook) {
  if (!self->grad_fn_node()) { JUST(AddAccumulateFunctionNode(self)); }
  self->mut_autograd_meta()->add_hook(hook);
  return Maybe<void>::Ok();
}
void ApiRegisterTensorHook(const std::shared_ptr<Tensor>& self, const AutogradMeta::Hook& hook) {
  return RegisterTensorHook(self, hook).GetOrThrow();
}

}  // namespace

ONEFLOW_API_PYBIND11_MODULE("", m) {
  py::class_<Tensor, std::shared_ptr<Tensor>>(m, "Tensor")
      .def(py::init(&MakeLocalTensor))
      .def(py::init(&MakeConsistentTensor))
      // Properties of pytorch
      .def_property_readonly("shape", &Tensor::shape)
      .def_property_readonly("dtype", &GetTensorDType)
      .def_property_readonly("is_cuda", &Tensor::is_cuda)
      .def_property_readonly("grad",
                             [](const Tensor& t) -> std::shared_ptr<Tensor> {
                               if (t.has_autograd_meta()) {
                                 return t.acc_grad().GetPtrOrThrow();
                               } else {
                                 return std::shared_ptr<Tensor>();
                               }
                             })
      // setter of grad
      .def("set_grad",
           [](Tensor& t, const std::shared_ptr<Tensor>& grad) {
             if (t.is_leaf()) {
               t.set_acc_grad(grad).GetOrThrow();
             } else {
               throw std::runtime_error("You can only change gradient of leaf tensors.");
             }
           })
      .def_property_readonly("grad_fn", &Tensor::grad_fn_node)
      .def_property_readonly("is_leaf", &Tensor::is_leaf)
      .def_property(
          "requires_grad", &Tensor::requires_grad,
          [](Tensor& t, bool requires_grad) {
            if (t.is_leaf()) {
              t.set_requires_grad(requires_grad);
            } else {
              throw std::runtime_error("You can only change requires_grad flags of leaf tensors.");
            }
          })
      // Methods of pytorch
      .def("retain_grad",
           [](Tensor& t) {
             if (!t.is_leaf()) { t.set_retain_grad(true).GetOrThrow(); }
           })
      .def("detach", [](const Tensor& t) { return t.detach().GetPtrOrThrow(); })
      .def("clone", [](const Tensor& t) { return t.clone().GetPtrOrThrow(); })
      // OneFlow tensor properties other than pytorch tensor
      .def_property_readonly("is_lazy", &Tensor::is_lazy)
      .def_property_readonly("is_eager", &Tensor::is_eager)
      .def_property_readonly("is_consistent", &Tensor::is_consistent)
      .def_property_readonly("is_local", &Tensor::is_local)
      .def("zeros_", &ApiEagerMirroredTensorZeros)
      .def("_register_hook", &ApiRegisterTensorHook)
      // local tensor only
      .def_property_readonly("_tensor_buffer_shapes_and_dtypes", &GetTensorBufferShapesAndDTypes)
      .def_property_readonly("device", &TensorGetDevice)
      .def_property_readonly("data", &Tensor::data)
#define DEFINE_TENSOR_METHOD(T, type_proto)                    \
  .def("_copy_to_numpy_" #T, &ApiCopyMirroredTensorToNumpy<T>) \
      .def("_copy_from_numpy_" #T, &ApiCopyMirroredTensorFromNumpy<T>)
          OF_PP_FOR_EACH_TUPLE(DEFINE_TENSOR_METHOD, POD_DATA_TYPE_SEQ)
#undef DEFINE_TENSOR_METHOD
      .def("_get_copy_mirrored_tensor_to_numpy_func_name", &ApiGetCopyMirroredTensorToNumpyFuncName)
      .def("_get_copy_mirrored_tensor_from_numpy_func_name",
           &ApiGetCopyMirroredTensorFromNumpyFuncName)
      // consistent tensor only
      .def_property_readonly("placement", &TensorGetParallelDesc);
}

}  // namespace one

}  // namespace oneflow
