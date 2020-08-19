
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
#include "oneflow/core/framework/framework.h"

extern "C" {
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
}

namespace oneflow {

namespace {
template<typename T>
PyArrayObject* vector_to_nparray(const std::vector<T>& vec, int type_num = NPY_FLOAT) {
  if (!vec.empty()) {
    size_t nRows = vec.size();
    npy_intp dims[0] = {nRows};

    PyArrayObject* vec_array = (PyArrayObject*)PyArray_SimpleNew(0, dims, type_num);
    T* vec_array_pointer = (T*)PyArray_DATA(vec_array);

    copy(vec.begin(), vec.end(), vec_array_pointer);
    return vec_array;
  } else {
    // no data at all
    npy_intp dims[0] = {0};
    return (PyArrayObject*)PyArray_ZEROS(0, dims, type_num, 0);
  }
}
}  // namespace

template<typename T>
class PyKernel : public user_op::OpKernel {
 public:
  PyKernel() = default;
  ~PyKernel() = default;

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    // size_t in_num = ctx->inputs().size();

    const T* in_dptrs = ctx->Tensor4ArgNameAndIndex("in", 0)->dptr<T>();
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    int64_t n = out->shape().elem_cnt();
    T* out_dptr = out->mut_dptr<T>();

    if (!PyEval_ThreadsInitialized()) { PyEval_InitThreads(); }
    PyGILState_STATE py_gil_st = PyGILState_Ensure();

    PyRun_SimpleString("print('hello')");

    PyObject *p_name, *p_module, *p_func;
    PyObject *p_args, *p_value;

    // load python kernel
    p_name = PyUnicode_DecodeFSDefault("pyk_sigmoid");
    // Error checking of pName left out
    p_module = PyImport_Import(p_name);
    Py_DECREF(p_name);
    if (p_module == nullptr) { PyErr_Print(); }

    // get forward func
    p_func = PyObject_GetAttrString(p_module, "forward");
    if (p_func == nullptr || !PyCallable_Check(p_func)) {
      Py_DECREF(p_module);
      PyErr_Print();
    }

    // input
    int num_input = 1;
    p_args = PyTuple_New(num_input);
    for (int i = 0; i < num_input; ++i) {
      p_value = PyLong_FromLong(1);
      if (p_value == nullptr) {
        Py_DECREF(p_args);
        Py_DECREF(p_module);
        CHECK(false) << "py_kernel cannot convert argument";
      }
      /* p_value reference stolen here: */
      PyTuple_SetItem(p_args, i, p_value);
    }

    // call func
    p_value = PyObject_CallObject(p_func, p_args);
    Py_DECREF(p_args);

    // output
    if (p_value != nullptr) {
      // deal with p_value
      Py_DECREF(p_value);
    } else {
      Py_DECREF(p_func);
      Py_DECREF(p_module);
      PyErr_Print();
    }
    Py_XDECREF(p_func);
    Py_DECREF(p_module);

    // dummy code to pass sigmoid test
    // for (int i = 0; i < n; ++i) { out_dptr[i] = 0.7310586; }

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
