#include "oneflow/core/register/register_desc.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/graph/task_node.h"
#include "oneflow/core/job/id_manager.h"

namespace oneflow {

RegstDesc::RegstDesc() {
  regst_desc_id_ = IDMgr::Singleton()->NewRegstDescId();
  is_blob_desc_locked_ = false;
}

void RegstDesc::AddConsumer(const TaskNode* new_consumer) {
  CHECK(consumers_.insert(new_consumer).second);
}

void RegstDesc::CopyBlobDescFrom(const RegstDesc* rhs) {
  CHECK_EQ(is_blob_desc_locked_, false);
  CHECK(lbn2blob_desc_.empty());
  for (const auto& pair : rhs->lbn2blob_desc_) {
    const std::string& lbn = pair.first;
    auto blob_desc = of_make_unique<BlobDesc>(*(pair.second));
    CHECK(lbn2blob_desc_.emplace(lbn, std::move(blob_desc)).second);
  }
}

BlobDesc* RegstDesc::AddLbn(const std::string& lbn) {
  CHECK_EQ(is_blob_desc_locked_, false);
  CHECK(lbn2blob_desc_.find(lbn) == lbn2blob_desc_.end()) << lbn;
  BlobDesc* blob_desc = new BlobDesc;
  lbn2blob_desc_[lbn].reset(blob_desc);
  return blob_desc;
}

const BlobDesc& RegstDesc::GetBlobDesc(const std::string& lbn) const {
  return *(lbn2blob_desc_.at(lbn));
}

BlobDesc* RegstDesc::MutBlobDesc(const std::string& lbn) {
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
        return it->second->ByteSizeOfDataField() == 0;
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
    ret->mutable_lbn2blob_desc()->insert(pb_pair);
  }
  ret->set_register_num(min_register_num_);
  ret->set_min_register_num(min_register_num_);
  ret->set_max_register_num(max_register_num_);
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
