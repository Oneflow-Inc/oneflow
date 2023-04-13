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
#include <memory>
#include <utility>
#include <vector>
#include "oneflow/api/python/of_api_registry.h"
#include "oneflow/api/python/job_build/job_build_and_infer.h"
#include "oneflow/core/common/throw.h"
#include "oneflow/core/framework/dtype.h"
#include "oneflow/core/framework/scope_util.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/core/autograd/autograd_engine.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/functional/functional.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/container_util.h"
#include "oneflow/core/framework/saved_tensor_hooks.h"
#include "oneflow/extension/stack/python/stack_getter.h"

namespace oneflow {
namespace autograd {

namespace {

bool IsScalarTensor(const one::Tensor& tensor) {
  const auto& shape = tensor.shape();
  return shape->elem_cnt() == 1;
}

// Checks and sets default value for initial gradients based on out_grads
// If output is the tensor whose size is greater than 1, out_grad's shape must be same as output's.
// If output is a scalar tensor, out_grad will also be a scaler or empty(will be initted to
// `oneflow.ones([1])`).
Maybe<one::TensorTuple> CheckAndInitOutGrads(const one::TensorTuple& outputs,
                                             const one::TensorTuple& out_grads) {
  size_t grad_size = out_grads.empty() ? outputs.size() : out_grads.size();
  auto gradients = std::make_shared<one::TensorTuple>(grad_size);
  CHECK_EQ_OR_RETURN(outputs.size(), gradients->size())
      << "RuntimeError: got " << outputs.size() << " tensors and " << gradients->size()
      << " gradients";
  for (int i = 0; i < outputs.size(); ++i) {
    CHECK_OR_RETURN(outputs.at(i)->requires_grad())
        << "\nRuntimeError: element " << i
        << " of tensors does not require grad and does not have a grad_fn";
    if (!outputs.at(i)->grad_fn_node()) {
      CHECK_OR_RETURN(outputs.at(i)->is_leaf())
          << "output[" << i << "] doesn't have grad_fn and it is not leaf tensor!\n"
          << "It is a bug with oneflow, please submit an issue on GitHub: "
             "https://github.com/Oneflow-Inc/oneflow/issues";
      JUST(one::AddAccumulateFunctionNode(outputs.at(i)));
    }
    if (out_grads.empty()) {
      CHECK_OR_RETURN(IsScalarTensor(*outputs.at(i)))
          << "Grad can be implicitly created only for scalar outputs";
      gradients->at(i) = JUST(one::functional::OnesLike(outputs.at(i)));
    } else {
      CHECK_OR_RETURN(*(outputs.at(i)->shape()) == *(out_grads.at(i)->shape()))
          << "out_grad's shape must be same as output's (" << outputs.at(i)->shape()->ToString()
          << " vs " << out_grads.at(i)->shape()->ToString() << ")";
      if (JUST(oneflow::VectorAt(outputs, i))->dtype()
          != JUST(oneflow::VectorAt(out_grads, i))->dtype()) {
        JUST(oneflow::VectorAt(*gradients, i)) =
            JUST(one::functional::Cast(out_grads[i], outputs[i]->dtype(), /*pin_memory=*/false));
      } else {
        JUST(oneflow::VectorAt(*gradients, i)) = out_grads[i];
      }
    }
  }
  if (LazyMode::is_enabled()) { JUST(MarkOutputGradients(outputs, *gradients)); }
  return gradients;
}

}  // namespace

Maybe<one::TensorTuple> Backward(const one::TensorTuple& outputs, const one::TensorTuple& out_grads,
                                 bool retain_graph, bool create_graph) {
  PythonFrameGuard pf;
  BackwardPassScopeGuard backward_guard;
  if (create_graph) { retain_graph = true; }
  std::shared_ptr<one::TensorTuple> gradients = JUST(CheckAndInitOutGrads(outputs, out_grads));
  JUST(one::GetThreadLocalAutogradEngine()->RunBackwardAndSaveGrads4LeafTensorIf(
      outputs, *gradients, retain_graph, create_graph));
  return std::make_shared<one::TensorTuple>(0);
}

Maybe<one::TensorTuple> Grad(const one::TensorTuple& outputs, const one::TensorTuple& inputs,
                             const one::TensorTuple& out_grads, bool retain_graph,
                             bool create_graph) {
  PythonFrameGuard pf;
  BackwardPassScopeGuard backward_guard;
  if (create_graph) { retain_graph = true; }
  if (inputs.empty()) { return Backward(outputs, out_grads, retain_graph, create_graph); }
  CHECK_OR_RETURN(std::all_of(
      inputs.begin(), inputs.end(),
      [](const std::shared_ptr<one::Tensor>& tensor) { return tensor->requires_grad(); }))
      << "All input tensors `.requires_grad` should be true";
  std::shared_ptr<one::TensorTuple> gradients = JUST(CheckAndInitOutGrads(outputs, out_grads));
  return one::GetThreadLocalAutogradEngine()->RunBackwardAndReturnInputsTensorGradIf(
      outputs, inputs, *gradients, retain_graph, create_graph);
}

namespace py = pybind11;

class PySavedTensorHook final : public one::SavedTensorHook {
 public:
  PySavedTensorHook(const py::function& pack_hook, const py::function& unpack_hook)
      : pack_hook_(pack_hook), unpack_hook_(unpack_hook) {}

  void pack(const std::shared_ptr<one::Tensor>& tensor) {
    py::gil_scoped_acquire acquire;
    py::object packed = pack_hook_(tensor);
    data_ = packed.release().ptr();
  }
  std::shared_ptr<one::Tensor> unpack() {
    py::gil_scoped_acquire acquire;
    py::object obj = py::cast<py::object>(data_);
    py::object x = unpack_hook_(obj);
    std::shared_ptr<one::Tensor> tensor;
    try {
      tensor = py::cast<std::shared_ptr<one::Tensor>>(x);
    } catch (const py::cast_error& e) {
      THROW(RuntimeError) << "unpack_hook should return a Tensor, but got `"
                          << py::str(x.get_type()).cast<std::string>() << "` instead";
    }
    return tensor;
  }

 private:
  PyObject* data_ = nullptr;
  py::function pack_hook_;
  py::function unpack_hook_;
};

class PySavedTensorHookCreator final : public one::SavedTensorHookCreator {
 public:
  std::unique_ptr<one::SavedTensorHook> new_saved_tensor_hook() const override {
    if (hooks_.empty()) { return nullptr; }
    return std::make_unique<PySavedTensorHook>(hooks_.back().first, hooks_.back().second);
  }
  void append_new_hooks(const py::function& pack_hook, const py::function& unpack_hook) {
    hooks_.emplace_back(pack_hook, unpack_hook);
  }
  void pop_hooks() {
    CHECK_OR_THROW(!hooks_.empty()) << "pop_hooks should not be called when there are no hooks";
    hooks_.pop_back();
  }

 private:
  small_vector<std::pair<py::function, py::function>, 1> hooks_;
};

ONEFLOW_API_PYBIND11_MODULE("autograd", m) {
  m.def("backward", &Backward);
  m.def("grad", &Grad);
  m.def_submodule("graph")
      .def("register_saved_tensors_hook_manager",
           []() {
             Singleton<one::SavedTensorHookCreator>::SetAllocated(new PySavedTensorHookCreator());
           })
      .def("append_new_hooks",
           [](const py::function& pack_hook, const py::function& unpack_hook) {
             PySavedTensorHookCreator* creator = dynamic_cast<PySavedTensorHookCreator*>(
                 Singleton<one::SavedTensorHookCreator>::Get());
             CHECK_NOTNULL_OR_THROW(creator)
                 << "`register_saved_tensors_hook_manager` should be called "
                    "before calling `append_new_hooks`";
             creator->append_new_hooks(pack_hook, unpack_hook);
           })
      .def("pop_hooks", []() {
        PySavedTensorHookCreator* creator =
            dynamic_cast<PySavedTensorHookCreator*>(Singleton<one::SavedTensorHookCreator>::Get());
        CHECK_NOTNULL_OR_THROW(creator) << "`register_saved_tensors_hook_manager` should be called "
                                           "before calling `pop_hooks`";
        creator->pop_hooks();
      });
}

}  // namespace autograd

}  // namespace oneflow
