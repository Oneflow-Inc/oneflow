#ifndef ONEFLOW_API_PYTHON_FRAMEWORK_SIZE_H_
#define ONEFLOW_API_PYTHON_FRAMEWORK_SIZE_H_

#include <Python.h>
#include "oneflow/core/common/shape.h"

PyObject* TensorSize_New(const Shape& size);

#endif  // ONEFLOW_API_PYTHON_FRAMEWORK_SIZE_H_
