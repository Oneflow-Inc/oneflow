#include "oneflow/core/register/register_desc.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/graph/copy_task_node.h"
#include "oneflow/core/job/id_manager.h"

namespace oneflow {

RegstDesc::RegstDesc() {
  regst_desc_id_ = IDMgr::Singleton()->NewRegstDescId();
  is_locked_ = false;
}

void RegstDesc::AddConsumer(const TaskNode* new_consumer) {
  CHECK(consumers_.insert(new_consumer).second);
}

void RegstDesc::Lock() {
  CHECK_EQ(is_locked_, false);
  is_locked_ = true;
  auto it = lbn2blob_desc_.begin();
  packed_blob_desc_.reset(new BlobDesc);
  *packed_blob_desc_ = ComputePackedBlobDesc([&]() {
    const BlobDesc* ret = nullptr;
    if (it != lbn2blob_desc_.end()) {
      ret = it->second.get();
      ++it;
    }
    return ret;
  });
}

void RegstDesc::CopyBlobDescFrom(const RegstDesc* rhs) {
  CHECK_EQ(is_locked_, false);
  CHECK(lbn2blob_desc_.empty());
  for (const auto& pair : rhs->lbn2blob_desc_) {
    const std::string& lbn = pair.first;
    AddLbn(lbn);
  }
  CopyBlobDescWithoutAddLbn(rhs);
}

void RegstDesc::CopyBlobDescWithoutAddLbn(const RegstDesc* rhs) {
  CHECK_EQ(is_locked_, false);
  for (const auto& pair : lbn2blob_desc_) {
    *(pair.second) = *(rhs->lbn2blob_desc_.at(pair.first));
  }
}

BlobDesc* RegstDesc::AddLbn(const std::string& lbn) {
  CHECK_EQ(is_locked_, false);
  CHECK(lbn2blob_desc_.find(lbn) == lbn2blob_desc_.end()) << lbn;
  BlobDesc* blob_desc = new BlobDesc;
  lbn2blob_desc_[lbn].reset(blob_desc);
  return blob_desc;
}

const BlobDesc* RegstDesc::GetBlobDesc(const std::string& lbn) const {
  return const_cast<RegstDesc*>(this)->MutBlobDesc(lbn);
}

BlobDesc* RegstDesc::MutBlobDesc(const std::string& lbn) {
  if (lbn == kPackedBlobName) { return packed_blob_desc_.get(); }
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

static void SetHostPinnedMemoryAccordingToConsumers(
    const HashSet<const TaskNode*>& consumers, MemoryCase* mem_case) {
  for (const TaskNode* consumer : consumers) {
    if (consumer->GetTaskType() == kCopyCommNet) {
      mem_case->mutable_host_pinned_mem()->set_used_by_network(true);
    }
    if (consumer->GetTaskType() == kCopyHd) {
      mem_case->mutable_host_pinned_mem()->set_used_by_device(true);
    }
  }
}

void RegstDesc::InferMemCase() {
  int64_t thrd_id = producer_->thrd_id();
  if (auto cp_hd_producer = dynamic_cast<const CopyHdTaskNode*>(producer_)) {
    if (cp_hd_producer->copy_type() == CopyHdOpConf::H2D) {
      mem_case_.mutable_device_cuda_mem()->set_device_id(thrd_id);
    } else {
      mem_case_.mutable_host_pinned_mem()->set_used_by_device(true);
      SetHostPinnedMemoryAccordingToConsumers(consumers_, &mem_case_);
    }
  } else if (producer_->GetTaskType() == kCopyCommNet) {
    mem_case_.mutable_host_pinned_mem()->set_used_by_network(true);
    SetHostPinnedMemoryAccordingToConsumers(consumers_, &mem_case_);
  } else {
    if (producer_->device_type() == kGPU) {
      mem_case_.mutable_device_cuda_mem()->set_device_id(thrd_id);
    } else {
      mem_case_.mutable_host_pageable_mem();
      SetHostPinnedMemoryAccordingToConsumers(consumers_, &mem_case_);
    }
  }
}

void RegstDesc::EraseBlobDesc(const std::string& lbn) {
  auto it = lbn2blob_desc_.find(lbn);
  if (it != lbn2blob_desc_.end()) { lbn2blob_desc_.erase(it); }
}

void RegstDesc::EraseZeroSizeBlob() {
  EraseIf<std::string, std::unique_ptr<BlobDesc>>(
      &lbn2blob_desc_,
      [](HashMap<std::string, std::unique_ptr<BlobDesc>>::iterator it) {
        return it->second->ByteSizeOfDataContentField() == 0;
      });
}

void RegstDesc::ToProto(RegstDescProto* ret) const {
  ret->set_regst_desc_id(regst_desc_id_);
  ret->set_producer_task_id(producer_->task_id());
  for (const TaskNode* consumer : consumers_) {
    ret->add_consumer_task_id(consumer->task_id());
  }
  for (const auto& pair : lbn2blob_desc_) {
    PbMapPair<std::string, BlobDescProto> pb_pair(pair.first);
    pair.second->ToProto(&(pb_pair.second));
    CHECK(ret->mutable_lbn2blob_desc()->insert(pb_pair).second);
  }
  packed_blob_desc_->ToProto(ret->mutable_packed_blob_desc());
  ret->set_min_register_num(min_register_num_);
  ret->set_max_register_num(max_register_num_);
  ret->set_register_num(min_register_num_);
  *(ret->mutable_mem_case()) = mem_case_;
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
