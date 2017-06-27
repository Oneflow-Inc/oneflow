#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/common/protobuf.h"

namespace oneflow {

void JobDesc::InitFromJobConf(const JobConf& conf) {
  LOG(INFO) << "Read JobConf...";
  ParseProtoFromTextFile(conf.train_dlnet_conf_filepath(), &train_dlnet_conf_);
  ParseProtoFromTextFile(conf.resource_filepath(), &resource_);
  ParseProtoFromTextFile(conf.strategy_filepath(), &strategy_);
  md_load_snapshot_path_ = conf.model_load_snapshot_path();
  md_save_snapshots_path_ = conf.model_save_snapshots_path();
  piece_size_ = conf.piece_size();
  num_of_pieces_in_batch_ = conf.num_of_pieces_in_batch();
  is_train_ = conf.is_train();
  floating_point_type_ = conf.floating_point_type();
  num_of_batches_in_snapshot_ = conf.num_of_batches_in_snapshot();
  staleness_ = conf.staleness();
  total_batch_num_ = conf.total_batch_num();
}

void JobDesc::InitFromProto(const JobDescProto& proto) {
  LOG(INFO) << "Init JobDesc from Proto";
  train_dlnet_conf_ = proto.train_dlnet_conf();
  resource_ = proto.resource();
  strategy_ = proto.strategy();
  md_load_snapshot_path_ = proto.model_load_snapshot_path();
  md_save_snapshots_path_ = proto.model_save_snapshots_path();
  piece_size_ = proto.piece_size();
  num_of_pieces_in_batch_ = proto.num_of_pieces_in_batch();
  is_train_ = proto.is_train();
  floating_point_type_ = proto.floating_point_type();
  num_of_batches_in_snapshot_ = proto.num_of_batches_in_snapshot();
  staleness_ = proto.staleness();
  total_batch_num_ = proto.total_batch_num();
}

void JobDesc::ToProto(JobDescProto* proto) const {
  *(proto->mutable_train_dlnet_conf()) = train_dlnet_conf_;
  *(proto->mutable_resource()) = resource_;
  *(proto->mutable_strategy()) = strategy_;
  *(proto->mutable_model_load_snapshot_path()) = md_load_snapshot_path_;
  *(proto->mutable_model_save_snapshots_path()) = md_save_snapshots_path_;
  proto->set_piece_size(piece_size_);
  proto->set_num_of_pieces_in_batch(num_of_pieces_in_batch_);
  proto->set_is_train(is_train_);
  proto->set_floating_point_type(floating_point_type_);
  proto->set_num_of_batches_in_snapshot(num_of_batches_in_snapshot_);
  proto->set_staleness(staleness_);
  proto->set_total_batch_num(total_batch_num_);
}

} // namespace oneflow
