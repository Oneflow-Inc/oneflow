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
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include "oneflow/core/common/data_type.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/util.h"

namespace oneflow {
namespace {
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

template<typename T>
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
  void* data = static_cast<void*>(const_cast<T*>(tensor->dptr<T>()));
  auto* np_array =
      reinterpret_cast<PyArrayObject*>(PyArray_SimpleNewFromData(dim_size, dims, type_num, data));
  // Numpy will not release the data
  PyArray_CLEARFLAGS(np_array, NPY_ARRAY_OWNDATA);
  *arg_ptr = reinterpret_cast<PyObject*>(np_array);
}

template<typename T>
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

  void* array_data_void = PyArray_DATA(array);
  T* array_data = static_cast<T*>(array_data_void);
  FOR_RANGE(int64_t, i, 0, array_elem_cnt) { tensor->mut_dptr<T>()[i] = array_data[i]; }
}

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

  // get GIL
  PyGILState_STATE py_gil_st;
  py_gil_st = PyGILState_Ensure();
  // prepare for numpy c api
  if (PyArray_API == nullptr) { _import_array(); }

  PyObject *py_file_str, *py_module, *py_func;
  PyObject *py_inputs, *py_outputs;

  // load python kernel
  const std::string grad_suffix = "_grad";
  std::string py_file_name = op_type_name;
  if (op_type_name.size() > grad_suffix.size()
      && op_type_name.rfind(grad_suffix) == (op_type_name.size() - grad_suffix.size())) {
    py_file_name = op_type_name.substr(0, op_type_name.size() - grad_suffix.size());
  }
  PyObject* sys_path = PySys_GetObject("path");
  PyList_Append(sys_path, PyUnicode_FromString(("./" + py_file_name).c_str()));
  py_file_name += "_py_kernel";
  py_file_str = PyUnicode_DecodeFSDefault(py_file_name.c_str());
  py_module = PyImport_Import(py_file_str);
  Py_DECREF(py_file_str);
  if (py_module == nullptr) { PyErr_Print(); }

  // get func
  py_func = PyObject_GetAttrString(py_module, py_func_name.c_str());
  if (py_func == nullptr || !PyCallable_Check(py_func)) {
    Py_DECREF(py_module);
    PyErr_Print();
  }

  // get numpy input
  MakePyInputs<T>(op_def, ctx, &py_inputs);

  // call func
  py_outputs = PyEval_CallObject(py_func, py_inputs);
  Py_DECREF(py_inputs);

  // get numpy output
  GetPyOutputs<T>(op_def, ctx, py_outputs);

  Py_XDECREF(py_func);
  Py_DECREF(py_module);
  Py_DECREF(py_outputs);

  // release GIL
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

template<typename T>
class PyGradKernel final : public user_op::OpKernel {
 public:
  PyGradKernel() = default;
  ~PyGradKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override { PyCompute<T>(ctx, "backward"); }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

}  // namespace oneflow
