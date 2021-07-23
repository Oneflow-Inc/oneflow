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
#include "oneflow/api/python/common.h"
#include "oneflow/api/python/of_api_registry.h"
#include "oneflow/api/python/ofblob/ofblob.e.h"
#include "oneflow/core/common/container_util.h"
#include "oneflow/core/common/tensor_buffer.h"
#include "oneflow/core/control/global_process_ctx.h"
#include "oneflow/core/framework/instructions_builder.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/tensor_method.h"
#include "oneflow/core/framework/device.h"
#include "oneflow/core/framework/stride.h"
#include "oneflow/core/framework/py_distribute.h"
#include "oneflow/core/job/placement.cfg.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/job/sbp_parallel.h"
#include "oneflow/core/job/resource_desc.h"
#include "oneflow/core/framework/dtype.h"
#include "oneflow/core/autograd/autograd_engine.h"
#include "oneflow/core/autograd/autograd_meta.h"
#include "oneflow/core/framework/id_util.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/framework/session_util.h"
#include "oneflow/core/functional/functional.h"
#include "oneflow/core/autograd/autograd_mode.h"
#include "oneflow/core/framework/op_expr_helper.h"

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

Symbol<Device> TensorGetDevice(const Tensor& tensor) { return tensor.device().GetOrThrow(); }

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

Maybe<Tensor> SyncDataAndMetaInfo(const std::shared_ptr<Tensor>& tensor,
                                  const std::vector<Symbol<cfg::SbpParallel>>& sbp_parallels,
                                  Symbol<ParallelDesc> parallel_desc) {
  if (sbp_parallels.size() == 1) {
    const auto& sbp_parallel = sbp_parallels.at(0);
    if (sbp_parallel->has_split_parallel()) {
      return tensor;
    } else if (sbp_parallel->has_broadcast_parallel()) {
      if (parallel_desc->device_tag() == "gpu") {
        TensorTuple input_list;
        input_list.emplace_back(tensor);
        std::shared_ptr<TensorTuple> outputs = std::make_shared<TensorTuple>(1);
        std::shared_ptr<UserOpExpr> op_expr =
            JUST(op_expr_helper::EagerNcclBroadcast(parallel_desc, 0));
        auto interperter = JUST(one::OpInterpUtil::GetInterpreter());
        JUST(interperter->Apply(*op_expr, input_list, outputs.get(), AttrMap{}));
        return outputs->at(0);
      } else {
        OF_UNIMPLEMENTED();
      }
    } else if (sbp_parallel->has_partial_sum_parallel()) {
      if (GlobalProcessCtx::Rank() == 0) {
        const auto& out_tensor = JUST(tensor->detach());
        bool requires_grad = autograd::GradMode::is_enabled() && tensor->requires_grad();
        out_tensor->set_requires_grad(requires_grad);
        out_tensor->set_is_leaf(!requires_grad);
        return out_tensor;
      } else {
        return functional::ZerosLike(tensor);
      }
    } else {
      OF_UNIMPLEMENTED();
    }
  } else {
    OF_UNIMPLEMENTED();
  }
}

// used in mirrored_tensor.to_consistent(sbp, placement)
Maybe<Tensor> CastLocalToConsistent(const std::shared_ptr<Tensor>& tensor,
                                    const std::vector<Symbol<cfg::SbpParallel>>& sbp_parallels,
                                    Symbol<ParallelDesc> parallel_desc) {
  const auto& mirrored_tensor = std::dynamic_pointer_cast<MirroredTensor>(tensor);
  CHECK_NOTNULL_OR_RETURN(mirrored_tensor) << "local tensors supported only";
  CHECK_OR_RETURN(mirrored_tensor->is_eager()) << "eager tensors supported only";
  if (mirrored_tensor->is_cuda()) {
    CHECK_EQ_OR_RETURN(
        JUST(mirrored_tensor->device())->device_id(),
        GlobalProcessCtx::LocalRank() % (Global<ResourceDesc, ForEnv>::Get()->GpuDeviceNum()))
        << "tensor must be on default device of rank!";
  }
  std::shared_ptr<Tensor> synced_tensor =
      JUST(SyncDataAndMetaInfo(mirrored_tensor, sbp_parallels, parallel_desc));
  TensorTuple input_list;
  input_list.emplace_back(synced_tensor);
  std::shared_ptr<TensorTuple> outputs = std::make_shared<TensorTuple>(1);
  const auto& op_expr = JUST(CastToConsistentOpExpr::New(*JUST(UniqueStr("cast_to_consistent")),
                                                         sbp_parallels, parallel_desc));
  const auto& session = JUST(GetDefaultSession());
  session->PushMirroredStrategyEnabled(false);
  auto interperter = JUST(one::OpInterpUtil::GetInterpreter());
  JUST(interperter->Apply(*op_expr, input_list, outputs.get(), AttrMap{}));
  session->PopMirroredStrategyEnabled();
  return outputs->at(0);
}

// used consistent_tensor.to_local()
Maybe<Tensor> CastConsistentToLocal(const std::shared_ptr<Tensor>& tensor) {
  const auto& consistent_tensor = std::dynamic_pointer_cast<ConsistentTensor>(tensor);
  CHECK_NOTNULL_OR_RETURN(consistent_tensor) << "consistent tensors supported only";
  CHECK_OR_RETURN(consistent_tensor->is_eager()) << "eager tensors supported only";
  int64_t machine_id = 0;
  int64_t device_id = 0;
  const auto& parallel_desc = JUST(consistent_tensor->parallel_desc());
  GlobalProcessCtx::GetCurrentMachineIdAndDeviceId(&machine_id, &device_id);
  if (!parallel_desc->Containing(machine_id, device_id)) {
    // should return UndefinesdLocalTensor here, the impl of which need to be discussed
    return std::shared_ptr<Tensor>();
  }
  TensorTuple input_list;
  input_list.emplace_back(consistent_tensor);
  auto outputs = std::make_shared<one::TensorTuple>(1);
  const auto& parallel_distribution = JUST(consistent_tensor->parallel_distribution());
  const auto& op_expr = JUST(CastFromConsistentOpExpr::New(*JUST(UniqueStr("cast_from_consistent")),
                                                           *parallel_distribution, parallel_desc));
  const auto& session = JUST(GetDefaultSession());
  session->PushMirroredStrategyEnabled(false);
  auto interperter = JUST(one::OpInterpUtil::GetInterpreter());
  JUST(interperter->Apply(*op_expr, input_list, outputs.get(), AttrMap{}));
  session->PopMirroredStrategyEnabled();
  return outputs->at(0);
}

Maybe<Tensor> ConvertTensorDevice(const std::shared_ptr<Tensor>& tensor,
                                  const std::string& device_type, int64_t device_id) {
  return functional::Copy(tensor, device_type, device_id);
}

bool ApiIsContiguous(const std::shared_ptr<Tensor>& tensor) {
  return IsContiguous(tensor).GetOrThrow();
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
      .def("storage_offset", [](const Tensor& t) { return t.storage_offset().GetOrThrow(); })
      .def("stride",
           [](const Tensor& t) {
             const auto& stride = t.stride().GetPtrOrThrow()->StrideVec();
             return py::tuple(py::make_iterator(stride.begin(), stride.end()));
           })
      .def("is_contiguous", &ApiIsContiguous)
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
      .def_property_readonly("placement", &TensorGetParallelDesc)
      .def("to_consistent",
           [](const std::shared_ptr<Tensor>& tensor,
              const std::vector<Symbol<cfg::SbpParallel>>& sbp_parallels,
              Symbol<ParallelDesc> parallel_desc) -> std::shared_ptr<Tensor> {
             if (tensor->is_consistent()) {
               UNIMPLEMENTED();
             } else {
               return CastLocalToConsistent(tensor, sbp_parallels, parallel_desc).GetPtrOrThrow();
             }
           })
      .def("to_local",
           [](const std::shared_ptr<Tensor>& tensor) -> std::shared_ptr<Tensor> {
             return CastConsistentToLocal(tensor).GetPtrOrThrow();
           })
      .def("to",
           [](const std::shared_ptr<Tensor>& tensor,
              const std::string& type_and_id) -> std::shared_ptr<Tensor> {
             if (tensor->is_consistent()) {
               std::string type;
               int device_id = -1;
               ParsingDeviceTag(type_and_id, &type, &device_id).GetOrThrow();
               if (device_id == -1) {
                 if (type == "cpu") {
                   device_id = GlobalProcessCtx::LocalRank()
                               % Global<ResourceDesc, ForEnv>::Get()->CpuDeviceNum();
                 } else {
                   device_id = GlobalProcessCtx::LocalRank()
                               % Global<ResourceDesc, ForEnv>::Get()->GpuDeviceNum();
                 }
               }
               return ConvertTensorDevice(tensor, type, device_id).GetPtrOrThrow();
             } else {
               UNIMPLEMENTED();
             }
           })
      .def(
          "to",
          [](const std::shared_ptr<Tensor>& tensor,
             Symbol<Device> device) -> std::shared_ptr<Tensor> {
            return ConvertTensorDevice(tensor, device->type(), device->device_id()).GetPtrOrThrow();
          });
}

}  // namespace one

}  // namespace oneflow
