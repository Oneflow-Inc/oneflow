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
#include "oneflow/extension/python/py_compute.h"

#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include "oneflow/core/common/data_type.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/util.h"

namespace oneflow {
namespace pyext {

namespace {
static PyObject* py_kernels_dic = nullptr;

void OFDataTypeToNumpyType(DataType of_data_type, int* out_numpy_type) {
  switch (of_data_type) {
    case DataType::kFloat: *out_numpy_type = NPY_FLOAT32; break;
    case DataType::kDouble: *out_numpy_type = NPY_FLOAT64; break;
    case DataType::kInt8: *out_numpy_type = NPY_INT8; break;
    case DataType::kInt32: *out_numpy_type = NPY_INT32; break;
    case DataType::kInt64: *out_numpy_type = NPY_INT64; break;
    case DataType::kUInt8: *out_numpy_type = NPY_UINT8; break;
    case DataType::kFloat16: *out_numpy_type = NPY_FLOAT16; break;
    default:
      LOG(FATAL) << "OneFlow data type " << DataType_Name(of_data_type)
                 << " is not valid to Numpy data type.";
  }
}

void NumpyTypeToOFDataType(PyArrayObject* array, DataType* of_data_type) {
  int py_array_type = PyArray_TYPE(array);
  switch (py_array_type) {
    case NPY_FLOAT32: *of_data_type = DataType::kFloat; break;
    case NPY_FLOAT64: *of_data_type = DataType::kDouble; break;
    case NPY_INT8: *of_data_type = DataType::kInt8; break;
    case NPY_INT32: *of_data_type = DataType::kInt32; break;
    case NPY_INT64: *of_data_type = DataType::kInt64; break;
    case NPY_UINT8: *of_data_type = DataType::kUInt8; break;
    case NPY_FLOAT16: *of_data_type = DataType::kFloat16; break;
    default:
      LOG(FATAL) << "Numpy data type " << py_array_type << " is not valid to OneFlow data type.";
  }
}

#define TENSOR_MEM_CAST(dtype) static_cast<void*>(const_cast<dtype*>(tensor->dptr<dtype>()))

void* TensorToMem(const user_op::Tensor* tensor) {
  switch (tensor->data_type()) {
    case DataType::kFloat: return TENSOR_MEM_CAST(float);
    case DataType::kDouble: return TENSOR_MEM_CAST(double);
    case DataType::kInt8: return TENSOR_MEM_CAST(int8_t);
    case DataType::kInt32: return TENSOR_MEM_CAST(int32_t);
    case DataType::kInt64: return TENSOR_MEM_CAST(int64_t);
    case DataType::kUInt8: return TENSOR_MEM_CAST(uint8_t);
    case DataType::kFloat16: return TENSOR_MEM_CAST(float16);
    default:
      LOG(FATAL) << "OneFlow data type " << DataType_Name(tensor->data_type())
                 << " is not supported yet.";
      return nullptr;
  }
}

void TensorToNumpy(const user_op::Tensor* tensor, PyObject** arg_ptr) {
  if (tensor == nullptr) {
    Py_INCREF(Py_None);
    *arg_ptr = Py_None;
    return;
  }
  int type_num = -1;
  OFDataTypeToNumpyType(tensor->data_type(), &type_num);
  LOG(INFO) << "Tensor data type " << DataType_Name(tensor->data_type()) << " Numpy type "
            << type_num;
  int dim_size = tensor->shape().NumAxes();
  npy_intp dims[dim_size];
  FOR_RANGE(size_t, i, 0, dim_size) { dims[i] = tensor->shape().At(i); }

  void* data = TensorToMem(tensor);
  auto* np_array =
      reinterpret_cast<PyArrayObject*>(PyArray_SimpleNewFromData(dim_size, dims, type_num, data));
  // Numpy will not release the data
  PyArray_CLEARFLAGS(np_array, NPY_ARRAY_OWNDATA);
  *arg_ptr = reinterpret_cast<PyObject*>(np_array);
}

#define TENSOR_MEM_ASSIGN(dtype)                                                     \
  do {                                                                               \
    dtype* array_data = static_cast<dtype*>(array_data_ptr);                         \
    FOR_RANGE(int64_t, i, 0, size) { tensor->mut_dptr<dtype>()[i] = array_data[i]; } \
  } while (0)

void MemToTensor(void* array_data_ptr, const size_t size, user_op::Tensor* tensor) {
  switch (tensor->data_type()) {
    case DataType::kFloat: TENSOR_MEM_ASSIGN(float); break;
    case DataType::kDouble: TENSOR_MEM_ASSIGN(double); break;
    case DataType::kInt8: TENSOR_MEM_ASSIGN(int8_t); break;
    case DataType::kInt32: TENSOR_MEM_ASSIGN(int32_t); break;
    case DataType::kInt64: TENSOR_MEM_ASSIGN(int64_t); break;
    case DataType::kUInt8: TENSOR_MEM_ASSIGN(uint8_t); break;
    case DataType::kFloat16: TENSOR_MEM_ASSIGN(float16); break;
    default:
      LOG(FATAL) << "OneFlow data type " << DataType_Name(tensor->data_type())
                 << " is not supported yet.";
  }
}

void NumpyToTensor(PyObject* arg, user_op::Tensor* tensor) {
  PyObject* ro_array = PyArray_FromAny(arg, nullptr, 0, 0, NPY_ARRAY_CARRAY_RO, nullptr);
  // PyArray_FromAny has increased the reference count
  Py_DECREF(ro_array);
  PyArrayObject* array = reinterpret_cast<PyArrayObject*>(ro_array);

  DataType of_data_type = DataType::kFloat;
  NumpyTypeToOFDataType(array, &of_data_type);
  CHECK_EQ(of_data_type, tensor->data_type())
      << "Numpy to OneFlow data type " << DataType_Name(of_data_type)
      << " is not equal to OneFlow tensor data type " << DataType_Name(tensor->data_type());

  int64_t array_elem_cnt = 1;
  FOR_RANGE(int, i, 0, PyArray_NDIM(array)) { array_elem_cnt *= PyArray_SHAPE(array)[i]; }
  CHECK_EQ(array_elem_cnt, tensor->shape().elem_cnt())
      << "Numpy array element count " << array_elem_cnt
      << " is not equal to OneFlow tensor element count " << tensor->shape().elem_cnt();

  void* array_data_ptr = PyArray_DATA(array);
  MemToTensor(array_data_ptr, array_elem_cnt, tensor);
}

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
    TensorToNumpy(ctx->Tensor4ArgNameAndIndex(arg_name, index), &arg);
    arg = PyArray_Return(reinterpret_cast<PyArrayObject*>(arg));
    PyList_SetItem(py_list, i, arg);
  }
  *py_inputs = Py_BuildValue("(N)", py_list);
  CHECK(*py_inputs);
}

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
      NumpyToTensor(PyList_GetItem(py_outputs, i), ctx->Tensor4ArgNameAndIndex(arg_name, index));
    }
  } else if (PyArray_Check(py_outputs)) {
    const std::string& arg_name = ctx->outputs().at(0).first;
    LOG(INFO) << "output arg_name " << arg_name;
    int32_t index = 0;
    NumpyToTensor(py_outputs, ctx->Tensor4ArgNameAndIndex(arg_name, index));
  } else {
    LOG(FATAL) << "Unexpeted PyObject was returned: " << Py_TYPE(py_outputs)->tp_name;
  }
}

}  // namespace

void PyRegisterKernels(PyObject* py_kernels) {
  if (py_kernels_dic == nullptr) {
    py_kernels_dic = py_kernels;
    Py_INCREF(py_kernels_dic);
  } else {
    LOG(FATAL) << "RegisterPyKernels should only be call once.";
  }
}

void PyCompute(user_op::KernelComputeContext* ctx, const std::string& py_func_name) {
  const std::string& op_type_name = ctx->user_op_conf().op_type_name();
  const user_op::OpRegistryResult* val =
      user_op::UserOpRegistryMgr::Get().GetOpRegistryResult(op_type_name);
  CHECK(val) << "Op op_type_name " << op_type_name << " has no definition.";
  const UserOpDef& op_def = val->op_def;

  // get GIL
  PyGILState_STATE py_gil_st;
  py_gil_st = PyGILState_Ensure();
  // prepare for numpy c api
  if (PyArray_API == nullptr) { _import_array(); }

  PyObject *py_str, *py_module, *py_func;
  PyObject *py_inputs, *py_outputs;

  // get python kernel
  static const std::string forward_suffix = "_forward";
  static const std::string backward_suffix = "_backward";
  std::string op_module_name = op_type_name;
  if (op_type_name.size() > forward_suffix.size()
      && op_type_name.rfind(forward_suffix) == (op_type_name.size() - forward_suffix.size())) {
    op_module_name = op_type_name.substr(0, op_type_name.size() - forward_suffix.size());
  }
  if (op_type_name.size() > backward_suffix.size()
      && op_type_name.rfind(backward_suffix) == (op_type_name.size() - backward_suffix.size())) {
    op_module_name = op_type_name.substr(0, op_type_name.size() - backward_suffix.size());
  }
  py_str = PyUnicode_DecodeFSDefault(op_module_name.c_str());
  CHECK(py_kernels_dic) << "py_kernels_dic should not be nullptr.";
  py_module = PyDict_GetItem(py_kernels_dic, py_str);
  CHECK(py_module) << op_module_name << " has no python kernel.";
  Py_DECREF(py_str);

  // get func
  py_func = PyObject_GetAttrString(py_module, py_func_name.c_str());
  if (py_func == nullptr || !PyCallable_Check(py_func)) {
    Py_DECREF(py_module);
    PyErr_Print();
  }

  // get numpy input
  MakePyInputs(op_def, ctx, &py_inputs);

  // call func
  py_outputs = PyEval_CallObject(py_func, py_inputs);
  Py_DECREF(py_inputs);

  // get numpy output
  GetPyOutputs(op_def, ctx, py_outputs);

  Py_XDECREF(py_func);
  Py_DECREF(py_outputs);

  // release GIL
  PyGILState_Release(py_gil_st);
}

}  // namespace pyext
}  // namespace oneflow
