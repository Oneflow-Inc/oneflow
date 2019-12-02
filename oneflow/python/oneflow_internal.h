#include "oneflow/python/oneflow_internal_helper.h"

void RegisterWatcherOnlyOnce(oneflow::ForeignWatcher* watcher, std::string* error_str) {
  return oneflow::RegisterWatcherOnlyOnce(watcher).GetDataAndSerializedErrorProto(error_str);
}

bool IsOpTypeCaseCpuSupportOnly(int64_t op_type_case, std::string* error_str) {
  return oneflow::IsOpTypeCaseCpuSupportOnly(op_type_case)
      .GetDataAndSerializedErrorProto(error_str, false);
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

void LaunchJob(const std::shared_ptr<oneflow::ForeignJobInstance>& cb, std::string* error_str) {
  return oneflow::LaunchJob(cb).GetDataAndSerializedErrorProto(error_str);
}

long long DeviceType4DeviceTag(const std::string& device_tag, std::string* error_str) {
  return oneflow::GetDeviceType4DeviceTag(device_tag)
      .GetDataAndSerializedErrorProto(error_str,
                                      static_cast<long long>(oneflow::DeviceType::kInvalidDevice));
}

std::string GetMachine2DeviceIdListOFRecordFromParallelConf(const std::string& parallel_conf,
                                                            std::string* error_str) {
  return oneflow::GetSerializedMachineId2DeviceIdListOFRecord(parallel_conf)
      .GetDataAndSerializedErrorProto(error_str, "");
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

void OfBlob_CopyShapeToNumpy(uint64_t of_blob_ptr, int64_t* array, int size) {
  using namespace oneflow;
  auto* of_blob = reinterpret_cast<OfBlob*>(of_blob_ptr);
  return of_blob->CopyShapeTo(array, size);
}
