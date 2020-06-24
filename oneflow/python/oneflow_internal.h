#include <stdint.h>
#include "oneflow/python/oneflow_internal_helper.h"
#include "oneflow/core/job/resource_desc.h"

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
  return oneflow::CurrentResource().GetDataAndSerializedErrorProto(error_str, "");
}

void EnableEagerExecution(bool enable_eager_execution) {
  using namespace oneflow;
  *Global<bool, EagerExecutionOption>::Get() = enable_eager_execution;
}

bool EagerExecutionEnabled() {
  using namespace oneflow;
  return *Global<bool, EagerExecutionOption>::Get();
}

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
  return oneflow::GetSerializedInterUserJobInfo().GetDataAndSerializedErrorProto(error_str, "");
}

std::string GetSerializedJobSet(std::string* error_str) {
  return oneflow::GetSerializedJobSet().GetDataAndSerializedErrorProto(error_str, "");
}

std::string GetFunctionConfigDef(std::string* error_str) {
  return oneflow::GetFunctionConfigDef().GetDataAndSerializedErrorProto(error_str, "");
}

void LaunchJob(const std::shared_ptr<oneflow::ForeignJobInstance>& cb, std::string* error_str) {
  return oneflow::LaunchJob(cb).GetDataAndSerializedErrorProto(error_str);
}

long DeviceType4DeviceTag(const std::string& device_tag, std::string* error_str) {
  return oneflow::GetDeviceType4DeviceTag(device_tag)
      .GetDataAndSerializedErrorProto(error_str,
                                      static_cast<long>(oneflow::DeviceType::kInvalidDevice));
}

std::string GetMachine2DeviceIdListOFRecordFromParallelConf(const std::string& parallel_conf,
                                                            std::string* error_str) {
  return oneflow::GetSerializedMachineId2DeviceIdListOFRecord(parallel_conf)
      .GetDataAndSerializedErrorProto(error_str, "");
}

std::string CheckAndCompleteUserOpConf(const std::string& serialized_op_conf,
                                       std::string* error_str) {
  return oneflow::CheckAndCompleteUserOpConf(serialized_op_conf)
      .GetDataAndSerializedErrorProto(error_str, "");
}

long CurrentMachineId(std::string* error_str) {
  return oneflow::CurrentMachineId().GetDataAndSerializedErrorProto(error_str, 0LL);
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
