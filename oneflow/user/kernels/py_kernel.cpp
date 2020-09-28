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

namespace {
template<typename T>
void MakePyInputs(const UserOpDef& op_def, user_op::KernelComputeContext* ctx,
                  PyObject** py_inputs) {
  const size_t kernel_in_num = ctx->inputs().size();
  const size_t def_in_num = op_def.input_size();
  CHECK_EQ(kernel_in_num, def_in_num) << "kernel input num " << kernel_in_num
                                      << " not equal to definition input num " << def_in_num;
  PyObject* py_list = PyList_New(def_in_num);
  CHECK(py_list);

  FOR_RANGE(size_t, i, 0, def_in_num) {
    PyObject* arg = nullptr;
    const std::string& arg_name = op_def.input(i).name();
    LOG(INFO) << "input arg_name " << arg_name;
    // do not support multi input in one symbolic arg name
    int32_t index = 0;
    TensorToNumpy<T>(ctx->Tensor4ArgNameAndIndex(arg_name, index), &arg);
    arg = PyArray_Return(reinterpret_cast<PyArrayObject*>(arg));
    PyList_SetItem(py_list, i, arg);
  }
  *py_inputs = Py_BuildValue("(N)", py_list);
  CHECK(*py_inputs);
}

template<typename T>
void GetPyOutputs(const UserOpDef& op_def, user_op::KernelComputeContext* ctx,
                  PyObject* py_outputs) {
  const size_t kernel_out_num = ctx->outputs().size();
  const size_t def_out_num = op_def.output_size();
  CHECK_EQ(kernel_out_num, def_out_num) << "kernel output num " << kernel_out_num
                                        << " not equal to definition output num " << def_out_num;
  if (PyList_Check(py_outputs)) {
    FOR_RANGE(size_t, i, 0, def_out_num) {
      const std::string& arg_name = op_def.output(i).name();
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
void PyCompute(user_op::KernelComputeContext* ctx, const std::string& py_func_name) {
  const std::string& op_type_name = ctx->user_op_conf().op_type_name();
  const user_op::OpRegistryResult* val =
      user_op::UserOpRegistryMgr::Get().GetOpRegistryResult(op_type_name);
  CHECK(val) << "Op op_type_name " << op_type_name << " has no definition.";
  const UserOpDef& op_def = val->op_def;

  // if (!PyEval_ThreadsInitialized()) { PyEval_InitThreads(); }
  PyGILState_STATE py_gil_st;
  py_gil_st = PyGILState_Ensure();
  if (PyArray_API == nullptr) { _import_array(); }

  PyObject *py_file, *py_module, *py_func;
  PyObject *py_inputs, *py_outputs;

  // load python kernel
  const std::string& py_file_name = ctx->Attr<std::string>("py_file");
  py_file = PyUnicode_DecodeFSDefault(py_file_name.c_str());
  // Error checking of pName left out
  py_module = PyImport_Import(py_file);
  Py_DECREF(py_file);
  if (py_module == nullptr) { PyErr_Print(); }

  // get forward func
  py_func = PyObject_GetAttrString(py_module, py_func_name.c_str());
  if (py_func == nullptr || !PyCallable_Check(py_func)) {
    Py_DECREF(py_module);
    PyErr_Print();
  }

  // input
  MakePyInputs<T>(op_def, ctx, &py_inputs);

  // call func
  py_outputs = PyEval_CallObject(py_func, py_inputs);
  Py_DECREF(py_inputs);

  // output
  GetPyOutputs<T>(op_def, ctx, py_outputs);

  Py_DECREF(py_outputs);
  Py_XDECREF(py_func);
  Py_DECREF(py_module);

  PyGILState_Release(py_gil_st);
}

}  // namespace

template<typename T>
class PyKernel : public user_op::OpKernel {
 public:
  PyKernel() = default;
  ~PyKernel() = default;

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override { PyCompute<T>(ctx, "forward"); }
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
  void Compute(user_op::KernelComputeContext* ctx) const override { PyCompute<T>(ctx, "backward"); }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_PY_GRAD_KERNEL(cpp_type, dtype)                                     \
  REGISTER_USER_KERNEL("pyg").SetCreateFn<PyGradKernel<cpp_type>>().SetIsMatchedHob( \
      (user_op::HobDeviceTag() == "cpu") & (user_op::HobDataType("dx", 0) == dtype));

OF_PP_FOR_EACH_TUPLE(REGISTER_PY_GRAD_KERNEL, ARITHMETIC_DATA_TYPE_SEQ);

}  // namespace oneflow
