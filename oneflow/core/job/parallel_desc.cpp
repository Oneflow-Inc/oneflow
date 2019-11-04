#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

namespace {

void ResetDeviceTag(std::string* device_name, const std::string& device_tag) {
  int64_t mchn_id = 0;
  std::string _ = "";
  std::string device_id_str = "";
  ParseDeviceNameConf(*device_name, &mchn_id, &_, &device_id_str);
  *device_name = std::to_string(mchn_id) + ":" + device_tag + ":" + device_id_str;
}

}  // namespace

void ParseDeviceNameConf(const std::string& device_name, int64_t* mchn_id, std::string* device_tag,
                         std::string* device_id_str) {
  size_t second_delimiter_pos = device_name.rfind(":");
  CHECK_NE(second_delimiter_pos, std::string::npos);
  size_t first_delimiter_pos = device_name.rfind(":", second_delimiter_pos - 1);
  CHECK_NE(first_delimiter_pos, std::string::npos);
  *mchn_id = oneflow_cast<int64_t>(device_name.substr(0, first_delimiter_pos));
  *device_tag =
      device_name.substr(first_delimiter_pos + 1, second_delimiter_pos - first_delimiter_pos - 1);
  *device_id_str = device_name.substr(second_delimiter_pos + 1);
}

std::string DeviceTag4DeviceType(DeviceType device_type) {
  if (device_type == kCPU) { return "cpu"; }
  if (device_type == kGPU) { return "gpu"; }
  UNIMPLEMENTED();
  return "";
}

Maybe<DeviceType> DeviceType4DeviceTag(const std::string& device_tag) {
  if (device_tag == "cpu") { return DeviceType::kCPU; }
  if (device_tag == "gpu") { return DeviceType::kGPU; }
  return Error::DeviceTagNotFound() << "device tag `" << device_tag << "' not found";
}

ParallelDesc::ParallelDesc(const ParallelConf& user_conf) : parallel_conf_(user_conf) {
  HashSet<int64_t> machine_id_set;
  device_type_ = DeviceType::kInvalidDevice;
  for (const std::string& device_name : parallel_conf_.device_name()) {
    int64_t mchn_id;
    std::string device_tag;
    std::string device_id_str;
    ParseDeviceNameConf(device_name, &mchn_id, &device_tag, &device_id_str);
    machine_id_set.insert(mchn_id);
    if (device_tag == "cpu") {
      CHECK(device_type_ == DeviceType::kInvalidDevice || device_type_ == DeviceType::kCPU);
      device_type_ = DeviceType::kCPU;
    } else if (device_tag == "gpu") {
      CHECK(device_type_ == DeviceType::kInvalidDevice || device_type_ == DeviceType::kGPU);
      device_type_ = DeviceType::kGPU;
    } else {
      UNIMPLEMENTED();
    }
    if (machine_id_set.find(mchn_id) == machine_id_set.end()) {
      sorted_machine_ids_.push_back(mchn_id);
    }
    int64_t minus_pos = device_id_str.find("-");
    if (minus_pos == std::string::npos) {
      device_id_str = device_id_str + "-" + device_id_str;
      minus_pos = device_id_str.find("-");
    }
    int64_t min_id = oneflow_cast<int64_t>(device_id_str.substr(0, minus_pos));
    int64_t max_id = oneflow_cast<int64_t>(device_id_str.substr(minus_pos + 1));
    CHECK_LE(min_id, max_id);
    for (int64_t dev_phy_id = min_id; dev_phy_id <= max_id; ++dev_phy_id) {
      if (device_type_ == DeviceType::kGPU) {
        CHECK_LT(dev_phy_id, Global<ResourceDesc>::Get()->GpuDeviceNum());
      }
      machine_id2sorted_dev_phy_ids_[mchn_id].push_back(dev_phy_id);
    }
  }
  ClearUp();
  SanityCheck();
}

bool ParallelDesc::Equals(const ParallelDesc& rhs) const {
  return device_type_ == rhs.device_type_ && sorted_machine_ids_ == rhs.sorted_machine_ids_
         && machine_id2sorted_dev_phy_ids_ == rhs.machine_id2sorted_dev_phy_ids_;
}

bool ParallelDesc::EqualsIgnoringDeviceType(const ParallelDesc& rhs) const {
  return sorted_machine_ids_ == rhs.sorted_machine_ids_
         && machine_id2sorted_dev_phy_ids_ == rhs.machine_id2sorted_dev_phy_ids_;
}

void ParallelDesc::ClearUp() {
  EraseIf<int64_t, std::vector<int64_t>>(
      &machine_id2sorted_dev_phy_ids_,
      [](HashMap<int64_t, std::vector<int64_t>>::iterator it) { return it->second.empty(); });
  sorted_machine_ids_.clear();
  parallel_num_ = 0;
  for (auto& pair : machine_id2sorted_dev_phy_ids_) {
    sorted_machine_ids_.push_back(pair.first);
    SortAndRemoveDuplication(&(pair.second));
    parallel_num_ += pair.second.size();
  }
  SortAndRemoveDuplication(&sorted_machine_ids_);
  int64_t parallel_id = 0;
  for (int64_t machine_id : sorted_machine_ids_) {
    for (int64_t device_id : machine_id2sorted_dev_phy_ids_.at(machine_id)) {
      parallel_id2machine_id_[parallel_id] = machine_id;
      parallel_id2device_id_[parallel_id] = device_id;
      parallel_id += 1;
    }
  }
}

void ParallelDesc::set_device_type(DeviceType device_type) {
  if (device_type == device_type_) { return; }
  device_type_ = device_type;
  const std::string& tag = DeviceTag4DeviceType(device_type);
  FOR_RANGE(int64_t, i, 0, parallel_conf_.device_name_size()) {
    ResetDeviceTag(parallel_conf_.mutable_device_name(i), tag);
  }
}

void ParallelDesc::SanityCheck() {
  device_num_of_each_machine_ = -1;
  for (auto& pair : machine_id2sorted_dev_phy_ids_) {
    if (device_num_of_each_machine_ == -1) {
      device_num_of_each_machine_ = pair.second.size();
    } else {
      CHECK_EQ(device_num_of_each_machine_, pair.second.size());
    }
  }
}

int64_t ParallelDesc::MachineIdForParallelId(int64_t parallel_id) const {
  return parallel_id2machine_id_.at(parallel_id);
}

int64_t ParallelDesc::DeviceIdForParallelId(int64_t parallel_id) const {
  return parallel_id2device_id_.at(parallel_id);
}

std::tuple<int32_t, int32_t> GetPartIdAndPartNumFromParallelCtx(
    const ParallelContext* parallel_ctx) {
  return std::make_tuple(parallel_ctx->parallel_id(), parallel_ctx->parallel_num());
}

ParallelConf GenParallelConfOfCpuZeroOnMaster() {
  ParallelConf parallel_conf;
  parallel_conf.add_device_name("0:cpu:0");
  return parallel_conf;
}

ParallelConf GenParallelConfOfCpuZeroOnAllMachines() {
  ParallelConf parallel_conf;
  FOR_RANGE(int64_t, i, 0, Global<ResourceDesc>::Get()->TotalMachineNum()) {
    parallel_conf.add_device_name(std::to_string(i) + ":cpu:0");
  }
  return parallel_conf;
}

}  // namespace oneflow
