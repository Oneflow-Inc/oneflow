
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
#include "oneflow/core/common/tensor_numpy_converter.h"
#include "oneflow/core/framework/framework.h"

namespace oneflow {

template<typename T>
void MakePyInputs(user_op::KernelComputeContext* ctx, PyObject** py_inputs) {
  size_t in_num = ctx->inputs().size();
  LOG(INFO) << "input num " << in_num;
  PyObject* py_list = PyList_New(in_num);
  CHECK(py_list);

  FOR_RANGE(size_t, i, 0, in_num) {
    PyObject* arg = nullptr;
    const std::string& arg_name = ctx->inputs().at(i).first;
    LOG(INFO) << "input arg_name " << arg_name;
    int32_t index = 0;
    TensorToNumpy<T>(ctx->Tensor4ArgNameAndIndex(arg_name, index), &arg);
    arg = PyArray_Return(reinterpret_cast<PyArrayObject*>(arg));
    PyList_SetItem(py_list, i, arg);
  }
  *py_inputs = Py_BuildValue("(N)", py_list);
  CHECK(*py_inputs);
}

template<typename T>
void GetPyOutputs(user_op::KernelComputeContext* ctx, PyObject* py_outputs) {
  if (PyList_Check(py_outputs)) {
    size_t out_num = ctx->outputs().size();
    LOG(INFO) << "output num " << out_num;
    FOR_RANGE(size_t, i, 0, out_num) {
      const std::string& arg_name = ctx->outputs().at(i).first;
      LOG(INFO) << "output arg_name " << arg_name;
      int32_t index = 0;
      NumpyToTensor<T>(PyList_GetItem(py_outputs, i), ctx->Tensor4ArgNameAndIndex(arg_name, index));
    }
  } else if (PyArray_Check(py_outputs)) {
    const std::string& arg_name = ctx->outputs().at(0).first;
    LOG(INFO) << "output arg_name " << arg_name;
    int32_t index = 0;
    NumpyToTensor<T>(py_outputs, ctx->Tensor4ArgNameAndIndex(arg_name, index));
  } else {
    LOG(FATAL) << "Unexpeted PyObject was returned: " << Py_TYPE(py_outputs)->tp_name;
  }
}

template<typename T>
class PyKernel : public user_op::OpKernel {
 public:
  PyKernel() = default;
  ~PyKernel() = default;

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    // if (!PyEval_ThreadsInitialized()) { PyEval_InitThreads(); }
    PyGILState_STATE py_gil_st;
    py_gil_st = PyGILState_Ensure();
    if (PyArray_API == nullptr) { _import_array(); }

    PyObject *py_file, *py_module, *py_func;
    PyObject *py_inputs, *py_outputs;

    // load python kernel
    py_file = PyUnicode_DecodeFSDefault("pyk_sigmoid");
    // Error checking of pName left out
    py_module = PyImport_Import(py_file);
    Py_DECREF(py_file);
    if (py_module == nullptr) { PyErr_Print(); }

    // get forward func
    py_func = PyObject_GetAttrString(py_module, "forward");
    if (py_func == nullptr || !PyCallable_Check(py_func)) {
      Py_DECREF(py_module);
      PyErr_Print();
    }

    // input
    MakePyInputs<T>(ctx, &py_inputs);

    // call func
    py_outputs = PyEval_CallObject(py_func, py_inputs);
    Py_DECREF(py_inputs);

    // output
    GetPyOutputs<T>(ctx, py_outputs);

    Py_DECREF(py_outputs);
    Py_XDECREF(py_func);
    Py_DECREF(py_module);

    PyGILState_Release(py_gil_st);
  }
};  // namespace oneflow

#define REGISTER_PY_KERNEL(cpp_type, dtype)                                     \
  REGISTER_USER_KERNEL("py").SetCreateFn<PyKernel<cpp_type>>().SetIsMatchedHob( \
      (user_op::HobDeviceTag() == "cpu") & (user_op::HobDataType("in", 0) == dtype));

OF_PP_FOR_EACH_TUPLE(REGISTER_PY_KERNEL, ARITHMETIC_DATA_TYPE_SEQ);

template<typename T>
class PyGradKernel final : public user_op::OpKernel {
 public:
  PyGradKernel() = default;
  ~PyGradKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x_blob = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* y_blob = ctx->Tensor4ArgNameAndIndex("y", 0);
    const user_op::Tensor* dy_blob = ctx->Tensor4ArgNameAndIndex("dy", 0);
    user_op::Tensor* dx_blob = ctx->Tensor4ArgNameAndIndex("dx", 0);
    // TODO(strint) : compute backward with py
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_PY_GRAD_KERNEL(cpp_type, dtype)                                         \
  REGISTER_USER_KERNEL("py_grad").SetCreateFn<PyGradKernel<cpp_type>>().SetIsMatchedHob( \
      (user_op::HobDeviceTag() == "cpu") & (user_op::HobDataType("dx", 0) == dtype));

OF_PP_FOR_EACH_TUPLE(REGISTER_PY_GRAD_KERNEL, ARITHMETIC_DATA_TYPE_SEQ);

}  // namespace oneflow
