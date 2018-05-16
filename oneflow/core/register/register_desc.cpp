#include "oneflow/core/register/register_desc.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/graph/copy_task_node.h"
#include "oneflow/core/job/id_manager.h"

namespace oneflow {

namespace {

LogicalBlobId GenUnCloneLbi(const LogicalBlobId& lbi) {
  LogicalBlobId ret(lbi);
  ret.set_clone_id(-1);
  return ret;
}

}  // namespace

RegstDesc::RegstDesc() {
  regst_desc_id_ = Global<IDMgr>::Get()->NewRegstDescId();
  producer_ = nullptr;
  min_register_num_ = 1;
  max_register_num_ = kMaxRegisterNum;
  is_locked_ = false;
}

void RegstDesc::AddConsumer(const TaskNode* new_consumer) {
  CHECK(consumers_.insert(new_consumer).second);
}

void RegstDesc::UpdtMinRegstNumIfNeed(int32_t val) {
  CHECK_LE(val, max_register_num_);
  min_register_num_ = std::max(min_register_num_, val);
}
void RegstDesc::UpdtMaxRegstNumIfNeed(int32_t val) {
  CHECK_GE(val, min_register_num_);
  max_register_num_ = std::min(max_register_num_, val);
}

void RegstDesc::Lock() {
  CHECK_EQ(is_locked_, false);
  is_locked_ = true;
  auto it = lbi2blob_desc_.begin();
  packed_blob_desc_.reset(new BlobDesc);
  *packed_blob_desc_ = ComputePackedBlobDesc([&]() {
    const BlobDesc* ret = nullptr;
    if (it != lbi2blob_desc_.end()) {
      ret = it->second.get();
      ++it;
    }
    return ret;
  });
}

void RegstDesc::CopyBlobDescFrom(const RegstDesc* rhs) {
  CHECK_EQ(is_locked_, false);
  CHECK(lbi2blob_desc_.empty());
  for (const auto& pair : rhs->lbi2blob_desc_) {
    const LogicalBlobId& lbi = pair.first;
    AddLbi(lbi);
  }
  CopyBlobDescWithoutAddLbi(rhs);
}

void RegstDesc::CopyBlobDescWithoutAddLbi(const RegstDesc* rhs) {
  CHECK_EQ(is_locked_, false);
  for (const auto& pair : lbi2blob_desc_) {
    auto rhs_it = rhs->lbi2blob_desc_.find(pair.first);
    if (rhs_it == rhs->lbi2blob_desc_.end()) {
      auto un_clone_it = rhs->lbi2blob_desc_.find(GenUnCloneLbi(pair.first));
      if (un_clone_it != rhs->lbi2blob_desc_.end()) { *(pair.second) = *(un_clone_it->second); }
    } else {
      *(pair.second) = *(rhs_it->second);
    }
  }
}

BlobDesc* RegstDesc::AddLbi(const LogicalBlobId& lbi) {
  CHECK_EQ(is_locked_, false);
  CHECK(lbi2blob_desc_.find(lbi) == lbi2blob_desc_.end());
  BlobDesc* blob_desc = new BlobDesc;
  lbi2blob_desc_[lbi].reset(blob_desc);
  return blob_desc;
}

const BlobDesc* RegstDesc::GetBlobDesc(const LogicalBlobId& lbi) const {
  return const_cast<RegstDesc*>(this)->MutBlobDesc(lbi);
}

BlobDesc* RegstDesc::MutBlobDesc(const LogicalBlobId& lbi) {
  if (lbi.is_packed_id()) { return packed_blob_desc_.get(); }
  auto it = lbi2blob_desc_.find(lbi);
  if (it != lbi2blob_desc_.end()) {
    return it->second.get();
  } else {
    return nullptr;
  }
}

void RegstDesc::ForEachLbi(std::function<void(const LogicalBlobId&)> func) const {
  for (const auto& p : lbi2blob_desc_) { func(p.first); }
}

void RegstDesc::EraseZeroSizeBlob() {
  EraseIf<LogicalBlobId, std::unique_ptr<BlobDesc>>(
      &lbi2blob_desc_, [](HashMap<LogicalBlobId, std::unique_ptr<BlobDesc>>::iterator it) {
        return it->second->ByteSizeOfDataContentField() == 0;
      });
}

void RegstDesc::ToProto(RegstDescProto* ret) const {
  ret->set_regst_desc_id(regst_desc_id_);
  ret->set_producer_task_id(producer_->task_id());
  for (const TaskNode* consumer : consumers_) { ret->add_consumer_task_id(consumer->task_id()); }
  for (const auto& pair : lbi2blob_desc_) {
    LbiBlobDescPair* pb_pair = ret->mutable_lbi2blob_desc()->Add();
    *(pb_pair->mutable_lbi()) = pair.first;
    pair.second->ToProto(pb_pair->mutable_blob_desc());
  }
  packed_blob_desc_->ToProto(ret->mutable_packed_blob_desc());
  ret->set_min_register_num(min_register_num_);
  ret->set_max_register_num(max_register_num_);
  ret->set_register_num(min_register_num_);
  *(ret->mutable_mem_case()) = mem_case_;
}

bool RegstDesc::HasSameBlobDescs(const RegstDesc* rhs) {
  if (rhs->lbi2blob_desc_.size() != lbi2blob_desc_.size()) { return false; }
  for (const auto& pair : rhs->lbi2blob_desc_) {
    auto iter = lbi2blob_desc_.find(pair.first);
    if (iter == lbi2blob_desc_.end()) { return false; }
    if (!(*(pair.second.get()) == *(iter->second.get()))) { return false; }
  }
  return true;
}

}  // namespace oneflow
