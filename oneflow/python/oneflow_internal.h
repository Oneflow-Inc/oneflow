#include "oneflow/python/oneflow_internal_helper.h"

void RegisterWatcherOnlyOnce(oneflow::ForeignWatcher* watcher, std::string* error_str) {
  return oneflow::RegisterWatcherOnlyOnce(watcher).GetDataAndSerializedErrorProto(error_str);
}

bool IsOpTypeCaseCpuSupportOnly(int64_t op_type_case, std::string* error_str) {
  return oneflow::IsOpTypeCaseCpuSupportOnly(op_type_case)
      .GetDataAndSerializedErrorProto(error_str, false);
}

bool IsEnvironmentInited() {
  using namespace oneflow;
  return Global<EnvironmentObjectsScope>::Get() != nullptr;
}

void InitEnvironmentBySerializedConfigProto(const std::string& config_proto_str,
                                            std::string* error_str) {
  return oneflow::InitEnvironmentBySerializedConfigProto(config_proto_str)
      .GetDataAndSerializedErrorProto(error_str);
}

void InitGlobalOneflow(std::string* error_str) {
  return oneflow::InitGlobalOneflow().GetDataAndSerializedErrorProto(error_str);
}

std::string GetSerializedInterUserJobInfo(std::string* error_str) {
  return oneflow::GetSerializedInterUserJobInfo().GetDataAndSerializedErrorProto(error_str, "");
}

void LaunchJob(const std::shared_ptr<oneflow::ForeignJobInstance>& cb, std::string* error_str) {
  return oneflow::LaunchJob(cb).GetDataAndSerializedErrorProto(error_str);
}

void DestroyGlobalOneflow(std::string* error_str) {
  return oneflow::DestroyGlobalOneflow().GetDataAndSerializedErrorProto(error_str);
}

void DestroyGlobalEnvironment(std::string* error_str) {
  return oneflow::DestroyGlobalEnvironment().GetDataAndSerializedErrorProto(error_str);
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

void OfBlob_CopyShapeFromNumpy(uint64_t of_blob_ptr, int64_t* array, int size) {
  using namespace oneflow;
  auto* of_blob = reinterpret_cast<OfBlob*>(of_blob_ptr);
  return of_blob->CopyShapeFrom(array, size);
}

void OfBlob_CopyShapeToNumpy(uint64_t of_blob_ptr, int64_t* array, int size) {
  using namespace oneflow;
  auto* of_blob = reinterpret_cast<OfBlob*>(of_blob_ptr);
  return of_blob->CopyShapeTo(array, size);
}

size_t OfBlob_GetNumOfLoDLevels(uint64_t of_blob_ptr) {
  using namespace oneflow;
  auto* of_blob = reinterpret_cast<OfBlob*>(of_blob_ptr);
  return of_blob->num_of_lod_levels();
}

bool OfBlob_IsDynamic(uint64_t of_blob_ptr) {
  using namespace oneflow;
  auto* of_blob = reinterpret_cast<OfBlob*>(of_blob_ptr);
  return of_blob->is_dynamic();
}

std::string OfBlob_GetSerializedLoDTree(uint64_t of_blob_ptr) {
  using namespace oneflow;
  auto* of_blob = reinterpret_cast<OfBlob*>(of_blob_ptr);
  return PbMessage2TxtString(of_blob->GetLoDTree());
}

void OfBlob_SetSerializedLoDTree(uint64_t of_blob_ptr, const std::string& lod_tree_str) {
  using namespace oneflow;
  LoDTree lod_tree;
  CHECK(TxtString2PbMessage(lod_tree_str, &lod_tree));
  auto* of_blob = reinterpret_cast<OfBlob*>(of_blob_ptr);
  of_blob->SetLoDTree(lod_tree);
}
