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
#include "oneflow/core/control/global_process_ctx.h"
#include "oneflow/core/framework/instructions_builder.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/device.h"
#include "oneflow/core/framework/py_distribute.h"
#include "oneflow/core/job/placement.cfg.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/framework/dtype.h"
#include "oneflow/core/autograd/autograd_engine.h"
#include "oneflow/core/autograd/autograd_meta.h"
#include "oneflow/core/framework/id_util.h"
#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/op_interpreter.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/framework/session_util.h"

namespace py = pybind11;

namespace oneflow {

namespace one {

namespace {

template<typename T>
const DType* GetTensorDType(const T& tensor) {
  return DType::Get(tensor.dtype()).GetOrThrow().get();
}

template<typename T>
struct TensorExportUtil final {};

template<>
struct TensorExportUtil<MirroredTensor> final {
  static std::shared_ptr<MirroredTensor> MakeTensor(const std::shared_ptr<const Shape>& shape,
                                                    const DType* dtype,
                                                    const std::shared_ptr<const Device>& device,
                                                    bool is_lazy, bool requires_grad,
                                                    bool is_leaf) {
    return MirroredTensor::MakeTensor(shape, dtype->data_type(), device, is_lazy, requires_grad,
                                      is_leaf)
        .GetPtrOrThrow();
  }
};

template<>
struct TensorExportUtil<ConsistentTensor> final {
  static std::shared_ptr<ConsistentTensor> MakeTensor(
      const std::shared_ptr<const Shape>& shape, const DType* dtype,
      const std::shared_ptr<const cfg::ParallelDistribution>& parallel_distribution,
      const std::shared_ptr<const ParallelDesc>& parallel_desc, bool is_lazy, bool requires_grad,
      bool is_leaf) {
    return ConsistentTensor::MakeTensor(shape, dtype->data_type(), SymbolOf(*parallel_distribution),
                                        SymbolOf(*parallel_desc), is_lazy, requires_grad, is_leaf)
        .GetPtrOrThrow();
  }
};

namespace {

Maybe<void> EagerMirroredTensorZeros(const std::shared_ptr<MirroredTensor>& tensor) {
  JUST(PhysicalRun([&](InstructionsBuilder* builder) {
    builder->AccessBlobByCallback(
        tensor,
        [](uint64_t of_blob_ptr) {
          auto* of_blob = reinterpret_cast<OfBlob*>(of_blob_ptr);
          of_blob->AsyncAutoMemset(0);
        },
        "mut");
  }));

  return Maybe<void>::Ok();
}

void ApiEagerMirroredTensorZeros(const std::shared_ptr<MirroredTensor>& tensor) {
  return EagerMirroredTensorZeros(tensor).GetOrThrow();
}

template<typename T>
Maybe<void> CopyBetweenMirroredTensorAndNumpy(const std::shared_ptr<MirroredTensor>& tensor,
                                              py::array_t<T> array,
                                              void (*Copy)(uint64_t, py::array_t<T>),
                                              const std::string& modifier) {
  std::atomic<bool> synced(false);

  JUST(PhysicalRun([&](InstructionsBuilder* builder) {
    builder->AccessBlobByCallback(
        tensor,
        [&array, &synced, &Copy](uint64_t ofblob_ptr) {
          Copy(ofblob_ptr, array);
          synced = true;
        },
        modifier);
  }));

  Global<ForeignLockHelper>::Get()->WithScopedRelease([&synced]() { /* spin wait */
                                                                    while (!synced) {}
  });

  return Maybe<void>::Ok();
}

template<typename T>
void ApiCopyMirroredTensorToNumpy(const std::shared_ptr<MirroredTensor>& tensor,
                                  py::array_t<T> array) {
  return CopyBetweenMirroredTensorAndNumpy(tensor, array, OfBlob_CopyToBuffer, "const")
      .GetOrThrow();
}

template<typename T>
void ApiCopyMirroredTensorFromNumpy(const std::shared_ptr<MirroredTensor>& tensor,
                                    py::array_t<T> array) {
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

std::shared_ptr<const Device> TensorGetDevice(const MirroredTensor& tensor) {
  return tensor.device().GetPtrOrThrow();
}

std::shared_ptr<const ParallelDesc> TensorGetParallelDesc(const ConsistentTensor& tensor) {
  return tensor.parallel_desc().GetOrThrow().shared_from_symbol();
}

std::tuple<std::vector<Shape>, std::vector<const DType*>> GetTensorBufferShapesAndDTypes(
    const std::shared_ptr<MirroredTensor>& tensor) {
  std::vector<Shape> shapes;
  std::vector<const DType*> dtypes;
  std::atomic<bool> synced(false);

  PhysicalRun([&](InstructionsBuilder* builder) {
    builder->AccessBlobByCallback(
        tensor, [&synced](uint64_t of_blob_ptr) { synced = true; }, "const");
  });

  Global<ForeignLockHelper>::Get()->WithScopedRelease([&synced]() {
    while (!synced) {}
  });

  const Blob& blob = CHECK_JUST(tensor->eager_blob_object())->blob();
  const Shape& blob_shape = blob.static_shape();
  const auto* tensor_buffer_ptr = blob.dptr<TensorBuffer>();
  for (int64_t i = 0; i < blob_shape.elem_cnt(); ++i) {
    const TensorBuffer* tensor_buffer = tensor_buffer_ptr + i;
    shapes.push_back(tensor_buffer->shape());
    dtypes.push_back(DType::Get(tensor_buffer->data_type()).GetOrThrow().get());
  }

  return std::make_tuple(shapes, dtypes);
}

}  // namespace

void SpecializedDef(py::class_<MirroredTensor, Tensor, std::shared_ptr<MirroredTensor>>* api) {
  using T = MirroredTensor;
  api->def_property_readonly("device", &TensorGetDevice);
  api->def_property_readonly("data", &T::data);
  api->def_property_readonly("_tensor_buffer_shapes_and_dtypes", &GetTensorBufferShapesAndDTypes);
#define DEFINE_TENSOR_METHOD(T, type_proto)                         \
  api->def("_copy_to_numpy_" #T, &ApiCopyMirroredTensorToNumpy<T>); \
  api->def("_copy_from_numpy_" #T, &ApiCopyMirroredTensorFromNumpy<T>);
  OF_PP_FOR_EACH_TUPLE(DEFINE_TENSOR_METHOD, POD_DATA_TYPE_SEQ);

#undef DEFINE_TENSOR_METHOD
  api->def("_get_copy_mirrored_tensor_to_numpy_func_name",
           &ApiGetCopyMirroredTensorToNumpyFuncName);
  api->def("_get_copy_mirrored_tensor_from_numpy_func_name",
           &ApiGetCopyMirroredTensorFromNumpyFuncName);
  api->def("zeros_", &ApiEagerMirroredTensorZeros);
  api->def("_register_hook",
           [](const std::shared_ptr<MirroredTensor>& self, const AutogradMeta::Hook& hook) -> void {
             self->mut_autograd_meta()->add_hook(hook);
           });
}

void SpecializedDef(py::class_<ConsistentTensor, Tensor, std::shared_ptr<ConsistentTensor>>* api) {
  api->def_property_readonly("placement", &TensorGetParallelDesc);
}

template<typename T>
py::class_<T, Tensor, std::shared_ptr<T>> ExportTensor(py::module& m, const char* name) {
  py::class_<T, Tensor, std::shared_ptr<T>> tensor_api(m, name);
  tensor_api
      .def(py::init(&TensorExportUtil<T>::MakeTensor))
      // Properties of pytorch
      .def_property_readonly("shape", &T::shape)
      .def_property_readonly("dtype", &GetTensorDType<T>)
      .def_property_readonly("is_cuda", &T::is_cuda)
      .def_property_readonly("grad", [](const T& t) { return t.api_acc_grad().GetPtrOrThrow(); })
      .def_property_readonly("grad_fn", &T::grad_fn_node)
      .def_property_readonly("is_leaf", &T::is_leaf)
      .def_property(
          "requires_grad", &T::requires_grad,
          [](T& t, bool requires_grad) {
            if (t.is_leaf()) {
              t.set_requires_grad(requires_grad);
            } else {
              throw std::runtime_error("You can only change requires_grad flags of leaf tensors.");
            }
          })
      // Methods of pytorch
      .def("retain_grad",
           [](T& t) {
             if (!t.is_leaf()) { t.set_retain_grad(true); }
           })
      .def("detach", [](const T& t) { return t.api_detach().GetPtrOrThrow(); })
      // OneFlow tensor properties other than pytorch tensor
      .def_property_readonly("is_lazy", &T::is_lazy)
      .def_property_readonly("is_consistent", &T::is_consistent);
  SpecializedDef(&tensor_api);
  return tensor_api;
}

// used in mirrored_tensor.to(sbp, placement)
Maybe<ConsistentTensor> CastLocalToConsistent(
    const std::shared_ptr<MirroredTensor>& mirrored_tensor,
    const std::vector<std::string>& sbp_parallels,
    const std::shared_ptr<ParallelDesc>& parallel_desc) {
  TensorTuple input_list;
  input_list.emplace_back(mirrored_tensor);
  std::shared_ptr<TensorTuple> outputs = std::make_shared<TensorTuple>(1);
  const auto& op_expr = JUST(CastToConsistentOpExpr::New(*JUST(UniqueStr("cast_to_consistent")),
                                                         sbp_parallels, parallel_desc));
  const auto& session = JUST(GetDefaultSession());
  session->PushMirroredStrategyEnabled(false);
  auto interperter = JUST(one::OpInterpUtil::GetInterpreter());
  JUST(interperter->Apply(*op_expr, input_list, outputs.get(), AttrMap{}));
  session->PopMirroredStrategyEnabled();
  const auto& out_tensor = std::dynamic_pointer_cast<ConsistentTensor>(outputs->at(0));
  return out_tensor;
}

// used consistent_tensor.to_local()
Maybe<MirroredTensor> CastConsistentToLocal(
    const std::shared_ptr<ConsistentTensor>& consistent_tensor) {
  int64_t machine_id = 0;
  int64_t device_id = 0;
  const auto& parallel_desc = JUST(consistent_tensor->parallel_desc());
  GlobalProcessCtx::GetCurrentMachineIdAndDeviceId(&machine_id, &device_id);
  if (!parallel_desc->Containing(machine_id, device_id)) {
    // should return UndefinesdLocalTensor here, the impl of which need to be discussed
    return std::shared_ptr<MirroredTensor>();
  }
  TensorTuple input_list;
  input_list.emplace_back(consistent_tensor);
  auto outputs = std::make_shared<one::TensorTuple>(1);
  const auto& parallel_distribution = JUST(consistent_tensor->parallel_distribution());
  const auto& op_expr = JUST(CastFromConsistentOpExpr::New(*JUST(UniqueStr("cast_from_consistent")),
                                                           *parallel_distribution, *parallel_desc));
  const auto& session = JUST(GetDefaultSession());
  session->PushMirroredStrategyEnabled(false);
  auto interperter = JUST(one::OpInterpUtil::GetInterpreter());
  JUST(interperter->Apply(*op_expr, input_list, outputs.get(), AttrMap{}));
  session->PopMirroredStrategyEnabled();
  const auto& out_tensor = std::dynamic_pointer_cast<MirroredTensor>(outputs->at(0));
  return out_tensor;
}

// used in consistent_tensor.to(sbp)
Maybe<ConsistentTensor> CastParallelDistribution(
    const std::shared_ptr<ConsistentTensor>& consistent_tensor,
    const std::vector<std::string>& sbp_parallels) {
  TensorTuple input_list;
  input_list.emplace_back(consistent_tensor);
  auto outputs = std::make_shared<one::TensorTuple>(1);
  const auto& parallel_distribution_cast_op_expr =
      JUST(OpBuilder("hierarchical_parallel_cast", *JUST(UniqueStr("hierarchical_parallel_cast")))
               .Input("in")
               .Output("out")
               .Attr<std::vector<std::string>>("parallel_distribution", sbp_parallels)
               .Attr<std::string>("grad_mode", "restore")
               .Attr<std::vector<std::string>>("grad_parallel_distribution", sbp_parallels)
               .Build());
  const auto& session = JUST(GetDefaultSession());
  session->PushMirroredStrategyEnabled(false);
  auto interperter = JUST(one::OpInterpUtil::GetInterpreter());
  JUST(interperter->Apply(*parallel_distribution_cast_op_expr, input_list, outputs.get(),
                          AttrMap{}));
  session->PopMirroredStrategyEnabled();
  const auto& out_tensor = std::dynamic_pointer_cast<ConsistentTensor>(outputs->at(0));
  return out_tensor;
}

}  // namespace

ONEFLOW_API_PYBIND11_MODULE("", m) {
  py::class_<Tensor, std::shared_ptr<Tensor>>(m, "Tensor");
  // args of tensor.to(...) need to be discussed
  ExportTensor<MirroredTensor>(m, "LocalTensor")
      .def("to",
           [](const std::shared_ptr<MirroredTensor>& mirrored_tensor,
              const std::vector<std::string>& sbp_parallels,
              const std::shared_ptr<ParallelDesc>& parallel_desc)
               -> std::shared_ptr<ConsistentTensor> {
             return CastLocalToConsistent(mirrored_tensor, sbp_parallels, parallel_desc)
                 .GetPtrOrThrow();
           });
  ExportTensor<ConsistentTensor>(m, "ConsistentTensor")
      .def("to",
           [](const std::shared_ptr<ConsistentTensor>& mirrored_tensor,
              const std::vector<std::string>& sbp_parallels) -> std::shared_ptr<ConsistentTensor> {
             return CastParallelDistribution(mirrored_tensor, sbp_parallels).GetPtrOrThrow();
           })
      .def("to_local",
           [](const std::shared_ptr<ConsistentTensor>& consistent_tensor)
               -> std::shared_ptr<MirroredTensor> {
             return CastConsistentToLocal(consistent_tensor).GetPtrOrThrow();
           });
}

}  // namespace one

}  // namespace oneflow
