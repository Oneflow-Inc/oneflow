#include "dag/register_info.h"
#include <vector>
#include <string>
#include <memory>
#include "memory/mem_util.h"
#include "memory/memory_manager.h"

namespace oneflow {
RegisterInfo::RegisterInfo()
  : RegisterInfo(RegisterType::kDataType, DeviceType::kCPU, -1, false) {
}

RegisterInfo::RegisterInfo(int64_t group_id)
  : RegisterInfo(RegisterType::kDataType, DeviceType::kCPU, group_id, false) {
}

RegisterInfo::RegisterInfo(RegisterType register_type, DeviceType device_type,
  int64_t group_id, bool network) : register_type_(register_type),
  device_type_(device_type),
  unaligned_memory_needed_(0),
  unaligned_memory_needed_for_envelope_(0),
  unaligned_memory_needed_for_non_envelope_(0),
  aligned_memory_needed_(0),
  total_elem_num_(0),
  group_id_(group_id),
  network_(network),
  has_envelope_(false),
  is_finalized_(false)  {
}

void RegisterInfo::set_register_type(RegisterType type) {
  CHECK(!is_finalized_);
  register_type_ = type;
}
void RegisterInfo::set_device_type(DeviceType type) {
  CHECK(!is_finalized_);
  device_type_ = type;
}
void RegisterInfo::set_group_id(int64_t group_id) {
  CHECK(!is_finalized_);
  group_id_ = group_id;
}
void RegisterInfo::set_network(bool network) {
  CHECK(!is_finalized_);
  network_ = network;
}

RegisterType RegisterInfo::register_type() const {
  // CHECK(is_finalized_);
  return register_type_;
}
DeviceType RegisterInfo::device_type() const {
  // CHECK(is_finalized_);
  return device_type_;
}
int64_t RegisterInfo::group_id() const {
  // CHECK(is_finalized_);
  CHECK(group_id_ != -1);
  return group_id_;
}
bool RegisterInfo::network() const {
  // CHECK(is_finalized_);
  return network_;
}
bool RegisterInfo::has_envelope() const {
  CHECK(is_finalized_);
  return has_envelope_;
}

const Shape& RegisterInfo::GetBlobShape(const std::string& blob_name) const {
  CHECK(is_finalized_);
  auto it = blob_dict_.find(blob_name);
  CHECK(it != blob_dict_.end());
  return it->second;
}

int64_t RegisterInfo::aligned_memory_needed() const {
  CHECK(is_finalized_);
  return aligned_memory_needed_;
}

int64_t RegisterInfo::total_element_num() const {
  CHECK(is_finalized_);
  return total_elem_num_;
}

const std::vector<std::string>& RegisterInfo::GetBlobNamesInsideEnvelope() const {
  return blob_names_inside_envelope_;
}

const std::vector<std::string>& RegisterInfo::GetBlobNamesOutsideEnvelope() const {
  return blob_names_outside_envelope_;
}

std::vector<std::string> RegisterInfo::GetEnvelopeNames() const {
  return envelope_names_;
}

const std::vector<std::string>& RegisterInfo::GetAllBlobNames() const {
  CHECK(is_finalized_);
  return blob_names_;
}

Shape RegisterInfo::GetEnvelopeShape() const {
  CHECK(is_finalized_);
  CHECK(has_envelope_);
  return envelope_shape_;
}


void RegisterInfo::AddEmptyBlob(const std::string& blob_name,
  EnvelopeFlag flag) {
  CHECK(!is_finalized_);
  if (flag == EnvelopeFlag::kOnEnvelope) {
    envelope_names_.push_back(blob_name);
  } else {
    // We use task_blob_name, a same task_blob_name may be added for multiple
    // times. We avoid the duplication while adding a new blob.
    if (blob_dict_.count(blob_name) == 0) {
      if (flag == EnvelopeFlag::kInEnvelope) {
        blob_names_inside_envelope_.push_back(blob_name);
      } else if (flag == EnvelopeFlag::kOutEnvelope) {
        blob_names_outside_envelope_.push_back(blob_name);
      }
      blob_names_.push_back(blob_name);
      blob_name_set_.insert(blob_name);
    }
  }
}

void RegisterInfo::SetBlobShape(const std::string& blob_name, const Shape& shape) {
  CHECK(!is_finalized_);
  auto it = blob_name_set_.find(blob_name);
  CHECK(it != blob_name_set_.end());
  blob_dict_.insert({ blob_name, shape });
}

void RegisterInfo::Finalize() {
  CHECK(!is_finalized_);
  total_elem_num_ = 0;
  // Ensure the names for true |envelope_flag| are contiguous
  int32_t envelope_blob_num = blob_names_inside_envelope_.size();
  for (int32_t i = 0; i < envelope_blob_num; ++i) {
    blob_names_.push_back(blob_names_inside_envelope_[i]);
    auto blob_shape = GetBlobShape(blob_names_inside_envelope_[i]);
    total_elem_num_ += blob_shape.count();
    unaligned_memory_needed_for_envelope_ += blob_shape.count();
  }

  int32_t non_envelope_blob_num = blob_names_outside_envelope_.size();
  for (int32_t i = 0; i < non_envelope_blob_num; ++i) {
    blob_names_.push_back(blob_names_outside_envelope_[i]);
    auto blob_shape = GetBlobShape(blob_names_outside_envelope_[i]);
    total_elem_num_ += blob_shape.count();
    unaligned_memory_needed_for_non_envelope_ += blob_shape.count();
  }

  CHECK((envelope_names_.empty() && unaligned_memory_needed_for_envelope_ == 0)
    || (!envelope_names_.empty() && unaligned_memory_needed_for_envelope_));

  has_envelope_ = !envelope_names_.empty();

  unaligned_memory_needed_
    = unaligned_memory_needed_for_envelope_
    + unaligned_memory_needed_for_non_envelope_;

  aligned_memory_needed_ = AlignSize(unaligned_memory_needed_);

  if (unaligned_memory_needed_for_envelope_) {
    envelope_shape_.Reshape({ 1, unaligned_memory_needed_for_envelope_ });
  }
  is_finalized_ = true;
}
}  // namespace oneflow
