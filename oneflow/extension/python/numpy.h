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
#include "oneflow/core/common/data_type.h"

// PyArrayObject cannot be forward declared, or compile error will occur
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

namespace oneflow {

namespace numpy {

Maybe<int> OFDataTypeToNumpyType(DataType of_data_type);

Maybe<DataType> NumpyTypeToOFDataType(int np_array_type);

Maybe<DataType> GetOFDataTypeFromNpArray(PyArrayObject* array);

}  // namespace numpy
}  // namespace oneflow
