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
#include "oneflow/api/python/of_api_registry.h"
#include "oneflow/core/common/preprocessor.h"
#include "oneflow/core/common/data_type_seq.h"
#include "oneflow/api/python/ofblob/ofblob.h"
#include "oneflow/api/python/ofblob/ofblob.e.h"

namespace py = pybind11;

ONEFLOW_API_PYBIND11_MODULE("", m) {
  m.def("Ofblob_GetDataType", &Ofblob_GetDataType);
  m.def("OfBlob_NumAxes", &OfBlob_NumAxes);
  m.def("OfBlob_IsDynamic", &OfBlob_IsDynamic);

  m.def("OfBlob_IsTensorList", &OfBlob_IsTensorList);
  m.def("OfBlob_TotalNumOfTensors", &OfBlob_TotalNumOfTensors);
  m.def("OfBlob_NumOfTensorListSlices", &OfBlob_NumOfTensorListSlices);

  m.def("OfBlob_TensorIndex4SliceId", &OfBlob_TensorIndex4SliceId);
  m.def("OfBlob_AddTensorListSlice", &OfBlob_AddTensorListSlice);
  m.def("OfBlob_ResetTensorIterator", &OfBlob_ResetTensorIterator);

  m.def("OfBlob_IncTensorIterator", &OfBlob_IncTensorIterator);
  m.def("OfBlob_CurTensorIteratorEqEnd", &OfBlob_CurTensorIteratorEqEnd);
  m.def("OfBlob_ClearTensorLists", &OfBlob_ClearTensorLists);

  m.def("OfBlob_AddTensor", &OfBlob_AddTensor);
  m.def("OfBlob_CurMutTensorAvailable", &OfBlob_CurMutTensorAvailable);
  m.def("OfBlob_CopyShapeFromNumpy", &OfBlob_CopyShapeFromNumpy);

  m.def("OfBlob_CopyShapeToNumpy", &OfBlob_CopyShapeToNumpy);
  m.def("OfBlob_CopyStaticShapeTo", &OfBlob_CopyStaticShapeTo);
  m.def("OfBlob_CurTensorCopyShapeTo", &OfBlob_CurTensorCopyShapeTo);

  m.def("OfBlob_CurMutTensorCopyShapeFrom", &OfBlob_CurMutTensorCopyShapeFrom);
  m.def("Dtype_GetOfBlobCurTensorCopyToBufferFuncName",
        &Dtype_GetOfBlobCurTensorCopyToBufferFuncName);
  m.def("Dtype_GetOfBlobCurMutTensorCopyFromBufferFuncName",
        &Dtype_GetOfBlobCurMutTensorCopyFromBufferFuncName);
  m.def("Dtype_GetOfBlobStaticTensorCopyFromBufferFuncName",
        &Dtype_GetOfBlobStaticTensorCopyFromBufferFuncName);

#define EXPORT_COPY_DATA_API(T, type_proto)                                                     \
  m.def("OfBlob_CurTensorCopyToBuffer_" OF_PP_STRINGIZE(T), &OfBlob_CurTensorCopyToBuffer_##T); \
  m.def("OfBlob_CurMutTensorCopyFromBuffer_" OF_PP_STRINGIZE(T),                                \
        &OfBlob_CurMutTensorCopyFromBuffer_##T);                                                \
  m.def("OfBlob_StaticTensorCopyFromBuffer_" OF_PP_STRINGIZE(T),                                \
        &OfBlob_StaticTensorCopyFromBuffer_##T);

  OF_PP_FOR_EACH_TUPLE(EXPORT_COPY_DATA_API, POD_DATA_TYPE_SEQ);

#undef EXPORT_COPY_DATA_API
}
