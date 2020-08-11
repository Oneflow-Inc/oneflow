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

void RegisterForeignCallbackOnlyOnce(oneflow::ForeignCallback* callback, std::string* error_str) {
  return oneflow::RegisterForeignCallbackOnlyOnce(callback).GetDataAndSerializedErrorProto(
      error_str);
}

void RegisterWatcherOnlyOnce(oneflow::ForeignWatcher* watcher, std::string* error_str) {
  return oneflow::RegisterWatcherOnlyOnce(watcher).GetDataAndSerializedErrorProto(error_str);
}

bool IsOpTypeCaseCpuSupportOnly(int64_t op_type_case, std::string* error_str) {
  return oneflow::IsOpTypeCaseCpuSupportOnly(op_type_case)
      .GetDataAndSerializedErrorProto(error_str, false);
}

bool IsOpTypeNameCpuSupportOnly(const std::string& op_type_name, std::string* error_str) {
  return oneflow::IsOpTypeNameCpuSupportOnly(op_type_name)
      .GetDataAndSerializedErrorProto(error_str, false);
}

std::string CurrentResource(std::string* error_str) {
  return oneflow::CurrentResource().GetDataAndSerializedErrorProto(error_str, std::string(""));
}

std::string EnvResource(std::string* error_str) {
  return oneflow::EnvResource().GetDataAndSerializedErrorProto(error_str, std::string(""));
}

void EnableEagerEnvironment(bool enable_eager_execution) {
  using namespace oneflow;
  *Global<bool, EagerExecution>::Get() = enable_eager_execution;
}

bool EagerExecutionEnabled() { return oneflow::EagerExecutionEnabled(); }

bool IsEnvInited() {
  using namespace oneflow;
  return Global<EnvGlobalObjectsScope>::Get() != nullptr;
}

void InitEnv(const std::string& env_proto_str, std::string* error_str) {
  return oneflow::InitEnv(env_proto_str).GetDataAndSerializedErrorProto(error_str);
}

void DestroyEnv(std::string* error_str) {
  return oneflow::DestroyEnv().GetDataAndSerializedErrorProto(error_str);
}

bool IsSessionInited() {
  using namespace oneflow;
  return Global<SessionGlobalObjectsScope>::Get() != nullptr;
}

void InitGlobalSession(const std::string& config_proto_str, std::string* error_str) {
  using namespace oneflow;
  return InitGlobalSession(config_proto_str).GetDataAndSerializedErrorProto(error_str);
}

void DestroyGlobalSession(std::string* error_str) {
  return oneflow::DestroyGlobalSession().GetDataAndSerializedErrorProto(error_str);
}

void StartGlobalSession(std::string* error_str) {
  return oneflow::StartGlobalSession().GetDataAndSerializedErrorProto(error_str);
}

void StopGlobalSession(std::string* error_str) {
  return oneflow::StopGlobalSession().GetDataAndSerializedErrorProto(error_str);
}

std::string GetSerializedInterUserJobInfo(std::string* error_str) {
  return oneflow::GetSerializedInterUserJobInfo().GetDataAndSerializedErrorProto(error_str,
                                                                                 std::string(""));
}

std::string GetSerializedJobSet(std::string* error_str) {
  return oneflow::GetSerializedJobSet().GetDataAndSerializedErrorProto(error_str, std::string(""));
}

std::string GetSerializedStructureGraph(std::string* error_str) {
  return oneflow::GetSerializedStructureGraph().GetDataAndSerializedErrorProto(error_str,
                                                                               std::string(""));
}

std::string GetFunctionConfigDef(std::string* error_str) {
  return oneflow::GetFunctionConfigDef().GetDataAndSerializedErrorProto(error_str, std::string(""));
}

void LaunchJob(const std::shared_ptr<oneflow::ForeignJobInstance>& cb, std::string* error_str) {
  return oneflow::LaunchJob(cb).GetDataAndSerializedErrorProto(error_str);
}

std::string GetMachine2DeviceIdListOFRecordFromParallelConf(const std::string& parallel_conf,
                                                            std::string* error_str) {
  return oneflow::GetSerializedMachineId2DeviceIdListOFRecord(parallel_conf)
      .GetDataAndSerializedErrorProto(error_str, std::string(""));
}

long GetUserOpAttrType(const std::string& op_type_name, const std::string& attr_name,
                       std::string* error_str) {
  return oneflow::GetUserOpAttrType(op_type_name, attr_name)
      .GetDataAndSerializedErrorProto(error_str, 0LL);
}

std::string InferOpConf(const std::string& serialized_op_conf,
                        const std::string& serialized_op_input_signature, std::string* error_str) {
  return oneflow::InferOpConf(serialized_op_conf, serialized_op_input_signature)
      .GetDataAndSerializedErrorProto(error_str, std::string(""));
}

long GetOpParallelSymbolId(const std::string& serialized_op_conf, std::string* error_str) {
  return oneflow::GetOpParallelSymbolId(serialized_op_conf)
      .GetDataAndSerializedErrorProto(error_str, 0LL);
}

std::string CheckAndCompleteUserOpConf(const std::string& serialized_op_conf,
                                       std::string* error_str) {
  return oneflow::CheckAndCompleteUserOpConf(serialized_op_conf)
      .GetDataAndSerializedErrorProto(error_str, std::string(""));
}

void RunLogicalInstruction(const std::string& vm_instruction_list,
                           const std::string& eager_symbol_list_str, std::string* error_str) {
  return oneflow::RunLogicalInstruction(vm_instruction_list, eager_symbol_list_str)
      .GetDataAndSerializedErrorProto(error_str);
}

void RunPhysicalInstruction(const std::string& vm_instruction_list,
                            const std::string& eager_symbol_list_str, std::string* error_str) {
  return oneflow::RunPhysicalInstruction(vm_instruction_list, eager_symbol_list_str)
      .GetDataAndSerializedErrorProto(error_str);
}

long CurrentMachineId(std::string* error_str) {
  return oneflow::CurrentMachineId().GetDataAndSerializedErrorProto(error_str, 0LL);
}

long NewLogicalObjectId(std::string* error_str) {
  return oneflow::NewLogicalObjectId().GetDataAndSerializedErrorProto(error_str, 0LL);
}

long NewLogicalSymbolId(std::string* error_str) {
  return oneflow::NewLogicalSymbolId().GetDataAndSerializedErrorProto(error_str, 0LL);
}

long NewPhysicalObjectId(std::string* error_str) {
  return oneflow::NewPhysicalObjectId().GetDataAndSerializedErrorProto(error_str, 0LL);
}

long NewPhysicalSymbolId(std::string* error_str) {
  return oneflow::NewPhysicalSymbolId().GetDataAndSerializedErrorProto(error_str, 0LL);
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

void CacheInt8Calibration(std::string* error_str) {
  oneflow::CacheInt8Calibration().GetDataAndSerializedErrorProto(error_str);
}

void WriteInt8Calibration(const std::string& path, std::string* error_str) {
  oneflow::WriteInt8Calibration(path).GetDataAndSerializedErrorProto(error_str);
}
