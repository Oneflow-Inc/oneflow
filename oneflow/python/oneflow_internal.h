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
#include <stdint.h>
#include "oneflow/python/oneflow_internal_helper.h"
#include "oneflow/core/job/resource_desc.h"

void RegisterWatcherOnlyOnce(oneflow::ForeignWatcher* watcher, std::string* error_str) {
  return oneflow::RegisterWatcherOnlyOnce(watcher).GetDataAndSerializedErrorProto(error_str);
}

void LaunchJob(const std::shared_ptr<oneflow::ForeignJobInstance>& cb, std::string* error_str) {
  return oneflow::LaunchJob(cb).GetDataAndSerializedErrorProto(error_str);
}

int Ofblob_GetDataType(uint64_t of_blob_ptr) {
  using namespace oneflow;
  auto* of_blob = reinterpret_cast<OfBlob*>(of_blob_ptr);
  return of_blob->data_type();
}

size_t OfBlob_NumAxes(uint64_t of_blob_ptr) {
  using namespace oneflow;
  auto* of_blob = reinterpret_cast<OfBlob*>(of_blob_ptr);
  return of_blob->NumAxes();
}

void OfBlob_CopyShapeFromNumpy(uint64_t of_blob_ptr, long* array, int size) {
  using namespace oneflow;
  auto* of_blob = reinterpret_cast<OfBlob*>(of_blob_ptr);
  return of_blob->CopyShapeFrom(array, size);
}

void OfBlob_CopyShapeToNumpy(uint64_t of_blob_ptr, long* array, int size) {
  using namespace oneflow;
  auto* of_blob = reinterpret_cast<OfBlob*>(of_blob_ptr);
  return of_blob->CopyShapeTo(array, size);
}

bool OfBlob_IsDynamic(uint64_t of_blob_ptr) {
  using namespace oneflow;
  auto* of_blob = reinterpret_cast<OfBlob*>(of_blob_ptr);
  return of_blob->is_dynamic();
}

bool OfBlob_IsTensorList(uint64_t of_blob_ptr) {
  using namespace oneflow;
  auto* of_blob = reinterpret_cast<OfBlob*>(of_blob_ptr);
  return of_blob->is_tensor_list();
}

long OfBlob_TotalNumOfTensors(uint64_t of_blob_ptr) {
  using namespace oneflow;
  auto* of_blob = reinterpret_cast<OfBlob*>(of_blob_ptr);
  return of_blob->TotalNumOfTensors();
}

long OfBlob_NumOfTensorListSlices(uint64_t of_blob_ptr) {
  using namespace oneflow;
  auto* of_blob = reinterpret_cast<OfBlob*>(of_blob_ptr);
  return of_blob->NumOfTensorListSlices();
}

long OfBlob_TensorIndex4SliceId(uint64_t of_blob_ptr, int32_t slice_id) {
  using namespace oneflow;
  auto* of_blob = reinterpret_cast<OfBlob*>(of_blob_ptr);
  return of_blob->TensorIndex4SliceId(slice_id);
}

void OfBlob_AddTensorListSlice(uint64_t of_blob_ptr) {
  using namespace oneflow;
  auto* of_blob = reinterpret_cast<OfBlob*>(of_blob_ptr);
  return of_blob->AddTensorListSlice();
}

void OfBlob_ResetTensorIterator(uint64_t of_blob_ptr) {
  using namespace oneflow;
  auto* of_blob = reinterpret_cast<OfBlob*>(of_blob_ptr);
  return of_blob->ResetTensorIterator();
}

void OfBlob_IncTensorIterator(uint64_t of_blob_ptr) {
  using namespace oneflow;
  auto* of_blob = reinterpret_cast<OfBlob*>(of_blob_ptr);
  return of_blob->IncTensorIterator();
}

bool OfBlob_CurTensorIteratorEqEnd(uint64_t of_blob_ptr) {
  using namespace oneflow;
  auto* of_blob = reinterpret_cast<OfBlob*>(of_blob_ptr);
  return of_blob->CurTensorIteratorEqEnd();
}

void OfBlob_CopyStaticShapeTo(uint64_t of_blob_ptr, long* array, int size) {
  using namespace oneflow;
  auto* of_blob = reinterpret_cast<OfBlob*>(of_blob_ptr);
  return of_blob->CopyStaticShapeTo(array, size);
}

void OfBlob_CurTensorCopyShapeTo(uint64_t of_blob_ptr, long* array, int size) {
  using namespace oneflow;
  auto* of_blob = reinterpret_cast<OfBlob*>(of_blob_ptr);
  return of_blob->CurTensorCopyShapeTo(array, size);
}

void OfBlob_ClearTensorLists(uint64_t of_blob_ptr) {
  using namespace oneflow;
  auto* of_blob = reinterpret_cast<OfBlob*>(of_blob_ptr);
  return of_blob->ClearTensorLists();
}

void OfBlob_AddTensor(uint64_t of_blob_ptr) {
  using namespace oneflow;
  auto* of_blob = reinterpret_cast<OfBlob*>(of_blob_ptr);
  return of_blob->AddTensor();
}

bool OfBlob_CurMutTensorAvailable(uint64_t of_blob_ptr) {
  using namespace oneflow;
  auto* of_blob = reinterpret_cast<OfBlob*>(of_blob_ptr);
  return of_blob->CurMutTensorAvailable();
}

void OfBlob_CurMutTensorCopyShapeFrom(uint64_t of_blob_ptr, long* array, int size) {
  using namespace oneflow;
  auto* of_blob = reinterpret_cast<OfBlob*>(of_blob_ptr);
  return of_blob->CurMutTensorCopyShapeFrom(array, size);
}
