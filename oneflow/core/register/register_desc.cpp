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
  enable_mem_sharing_ = false;
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
  GenPackedBlobDesc();
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
  *(ret->mutable_regst_desc_type()) = regst_desc_type_;
  if (regst_desc_type_.has_data_regst_desc()) {
    DataRegstDesc* data_regst_desc_proto =
        ret->mutable_regst_desc_type()->mutable_data_regst_desc();
    packed_blob_desc_->ToProto(data_regst_desc_proto->mutable_packed_blob_desc());
    for (const auto& pair : lbi2blob_desc_) {
      LbiBlobDescPair* pb_pair = data_regst_desc_proto->mutable_lbi2blob_desc()->Add();
      *(pb_pair->mutable_lbi()) = pair.first;
      pair.second->ToProto(pb_pair->mutable_blob_desc());
    }
  } else if (regst_desc_type_.has_ctrl_regst_desc()) {
    // do nothing
  } else {
    UNIMPLEMENTED();
  }
  ret->set_min_register_num(min_register_num_);
  ret->set_max_register_num(max_register_num_);
  ret->set_register_num(min_register_num_);
  *(ret->mutable_mem_case()) = mem_case_;
  ret->set_enable_mem_sharing(enable_mem_sharing_);
  ret->set_mem_shared_id(-1);
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

void RegstDesc::GenPackedBlobDesc() {
  if (lbi2blob_desc_.size() == 1) {
    packed_blob_desc_.reset(new BlobDesc(*lbi2blob_desc_.begin()->second));
  } else if (lbi2blob_desc_.size() > 1) {
    int64_t total_byte_size = 0;
    int64_t total_data_byte_size = 0;
    HashSet<int> data_type_set;
    bool has_data_id_field = false;
    bool has_col_num_field = false;
    int32_t max_col_num = -1;
    for (const auto& pair : lbi2blob_desc_) {
      const BlobDesc& blob_desc = *pair.second;
      total_byte_size += blob_desc.TotalByteSize();
      total_data_byte_size += blob_desc.AlignSizeOfDataContentField();
      data_type_set.insert(static_cast<int>(blob_desc.data_type()));
      has_data_id_field = has_data_id_field || blob_desc.has_data_id_field();
      has_col_num_field = has_col_num_field || blob_desc.has_col_num_field();
      has_col_num_field = has_col_num_field || blob_desc.has_col_num_field();
      if (max_col_num == -1) {
        max_col_num = blob_desc.max_col_num();
      } else {
        CHECK_EQ(max_col_num, blob_desc.max_col_num());
      }
    }

    BlobDesc* packed_blob_desc = new BlobDesc;
    int64_t packed_blob_size = 0;
    bool packed_as_raw = false;
    if (mem_case_.has_host_mem() && mem_case_.host_mem().used_by_network()) {
      packed_blob_size = total_byte_size;
      packed_as_raw = has_data_id_field || has_col_num_field || (data_type_set.size() != 1);
    } else {
      packed_blob_size = total_data_byte_size;
      packed_as_raw = (data_type_set.size() != 1);
    }
    if (!packed_as_raw) {
      DataType sole_data_type = static_cast<DataType>(*(data_type_set.begin()));
      int64_t size_of_one_elem = GetSizeOfDataType(sole_data_type);
      CHECK_EQ(packed_blob_size % size_of_one_elem, 0);
      packed_blob_desc->mut_shape() = Shape({packed_blob_size / size_of_one_elem});
      packed_blob_desc->set_data_type(sole_data_type);
    } else {
      packed_blob_desc->mut_shape() = Shape({packed_blob_size});
      packed_blob_desc->set_data_type(DataType::kChar);
    }
    packed_blob_desc->set_has_data_id_field(false);
    packed_blob_desc->set_has_col_num_field(false);
    packed_blob_desc->set_max_col_num(1);
    packed_blob_desc_.reset(packed_blob_desc);
  } else {
    packed_blob_desc_.reset(new BlobDesc);
  }
}

void InitCtrlRegstDesc(int64_t producer_task_id, RegstDescProto* ctrl_regst_proto) {
  CHECK_NOTNULL(ctrl_regst_proto);
  ctrl_regst_proto->set_regst_desc_id(Global<IDMgr>::Get()->NewRegstDescId());
  ctrl_regst_proto->set_producer_task_id(producer_task_id);
  ctrl_regst_proto->set_min_register_num(1);
  ctrl_regst_proto->set_max_register_num(1);
  ctrl_regst_proto->set_register_num(1);
  ctrl_regst_proto->mutable_regst_desc_type()->mutable_ctrl_regst_desc();
  ctrl_regst_proto->mutable_mem_case()->mutable_host_mem();
  ctrl_regst_proto->set_enable_mem_sharing(false);
  ctrl_regst_proto->set_mem_shared_id(-1);
}

}  // namespace oneflow
