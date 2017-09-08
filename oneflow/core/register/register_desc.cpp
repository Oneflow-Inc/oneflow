#include "oneflow/core/register/register_desc.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/graph/copy_task_node.h"
#include "oneflow/core/graph/task_node.h"
#include "oneflow/core/job/id_manager.h"

namespace oneflow {

namespace {

void SetDeviceCudaMemoryAccordingToThrdLocId(MemoryCase& mem_case,
                                             int64_t thrd_loc_id) {
  int64_t device_id = IDMgr::Singleton()->DevPhyId4ThrdLocId(thrd_loc_id);
  mem_case.mutable_device_cuda_mem()->set_device_id(device_id);
}

void SetHostPinnedMemoryAccordingToConsumers(
    MemoryCase& mem_case, const HashSet<const TaskNode*>& subs) {
  for (const TaskNode* sub : subs) {
    if (sub->task_type() == kCopyCommNetTask) {
      mem_case.mutable_host_pinned_mem()->set_need_rdma(true);
    }
    if (auto cp_hd_sub = dynamic_cast<const CopyHDTaskNode*>(sub)) {
      if (cp_hd_sub->IsH2D()) {
        mem_case.mutable_host_pinned_mem()->set_need_cuda(true);
      }
    }
  }
}

}  // namespace

RegstDesc::RegstDesc() {
  producer_ = nullptr;
  register_num_ = 3;  // TODO
}

void RegstDesc::AddConsumer(const TaskNode* new_consumer) {
  CHECK(consumers_.insert(new_consumer).second);
}

void RegstDesc::CopyLbnFrom(const RegstDesc* rhs) {
  CHECK(lbn2blob_desc_.empty());
  for (const auto& pair : rhs->lbn2blob_desc_) {
    const std::string& lbn = pair.first;
    CHECK(lbn2blob_desc_.emplace(lbn, of_make_unique<BlobDesc>()).second);
  }
}

void RegstDesc::CopyBlobDescFrom(const RegstDesc* rhs) {
  for (const auto& pair : lbn2blob_desc_) {
    const std::string& lbn = pair.first;
    *(lbn2blob_desc_.at(lbn)) = rhs->GetBlobDesc(lbn);
  }
}

void RegstDesc::EnrollLbn(const std::string& lbn) {
  CHECK(lbn2blob_desc_.emplace(lbn, of_make_unique<BlobDesc>()).second) << lbn;
}

const BlobDesc& RegstDesc::GetBlobDesc(const std::string& lbn) const {
  return *(lbn2blob_desc_.at(lbn));
}

BlobDesc* RegstDesc::GetMutBlobDesc(const std::string& lbn) {
  auto it = lbn2blob_desc_.find(lbn);
  if (it != lbn2blob_desc_.end()) {
    return it->second.get();
  } else {
    return nullptr;
  }
}

void RegstDesc::ForEachLbn(std::function<void(const std::string&)> func) const {
  for (const auto& p : lbn2blob_desc_) { func(p.first); }
}

void RegstDesc::EraseZeroSizeBlob() {
  EraseIf<std::string, std::unique_ptr<BlobDesc>>(
      &lbn2blob_desc_,
      [](HashMap<std::string, std::unique_ptr<BlobDesc>>::iterator it) {
        return it->second->shape().elem_cnt() == 0;
      });
}

int64_t RegstDesc::CompElemCntOfAllBlob() const {
  int64_t sum = 0;
  for (const auto& pair : lbn2blob_desc_) {
    sum += pair.second->shape().elem_cnt();
  }
  return sum;
}

std::string RegstDesc::DebugStr() const {
  std::stringstream ss;
  ss << "{";
  for (const auto& pair : lbn2blob_desc_) {
    ss << "{" << pair.first << ":" << pair.second->shape().DebugStr() << "}";
  }
  ss << "}";
  return ss.str();
}

void RegstDesc::ToProto(RegstDescProto* ret) const {
  CHECK(min_register_num_ <= register_num_);
  CHECK(register_num_ <= max_register_num_);
  ret->set_regst_desc_id(regst_desc_id_);
  ret->set_producer_task_id(producer_->task_id());
  for (const TaskNode* consumer : consumers_) {
    if (!consumer->IsMeaningLess()) {
      ret->add_consumer_task_id(consumer->task_id());
    }
  }
  for (const auto& pair : lbn2blob_desc_) {
    PbMapPair<std::string, BlobDescProto> pb_pair(pair.first);
    pair.second->ToProto(&(pb_pair.second));
    ret->mutable_lbn2blob_desc()->insert(pb_pair);
  }
  ret->set_register_num(register_num_);
  ret->set_min_register_num(min_register_num_);
  ret->set_max_register_num(max_register_num_);
  *(ret->mutable_mem_case()) = InferMemCase();
}

MemoryCase RegstDesc::InferMemCase() const {
  MemoryCase mem_case;
  DeviceType device_type =
      producer_->chain_node()->parallel_desc()->device_type();
  if (auto cp_hd_producer = dynamic_cast<const CopyHDTaskNode*>(producer_)) {
    if (cp_hd_producer->IsH2D()) {
      SetDeviceCudaMemoryAccordingToThrdLocId(mem_case,
                                              producer_->thrd_loc_id());
    } else {
      mem_case.mutable_host_pinned_mem()->set_need_cuda(true);
      SetHostPinnedMemoryAccordingToConsumers(mem_case, consumers_);
    }
  } else if (producer_->task_type() == kCopyCommNetTask) {
    mem_case.mutable_host_pinned_mem()->set_need_rdma(true);
    SetHostPinnedMemoryAccordingToConsumers(mem_case, consumers_);
  } else {
    if (device_type == kGPU && producer_->task_type() != kBoxingTask) {
      SetDeviceCudaMemoryAccordingToThrdLocId(mem_case,
                                              producer_->thrd_loc_id());
    } else {
      mem_case.mutable_host_pageable_mem();
      SetHostPinnedMemoryAccordingToConsumers(mem_case, consumers_);
    }
  }
  return mem_case;
}

BlobDesc RegstDesc::CompPackedBlobDesc() const {
  auto it = lbn2blob_desc_.begin();
  return ComputePackedBlobDesc([&]() {
    const BlobDesc* ret = nullptr;
    if (it != lbn2blob_desc_.end()) {
      ret = it->second.get();
      ++it;
    }
    return ret;
  });
}

}  // namespace oneflow
