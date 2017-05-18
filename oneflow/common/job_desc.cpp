#include "common/job_desc.h"
#include "common/protobuf.h"

namespace oneflow {

void JobDesc::InitFromJobConf(const JobConf& conf) {
  LOG(INFO) << "Read JobConf...";
  ParseProtoFromTextFile(conf.train_dlnet_conf_filepath(), &train_dlnet_conf_);
  ParseProtoFromTextFile(conf.resource_filepath(), &resource_);
  ParseProtoFromTextFile(conf.strategy_filepath(), &strategy_);
  md_load_machine_ = conf.model_load_machine();
  md_save_machine_ = conf.model_save_machine();
  batch_size_ = conf.batch_size();
  piece_size_ = conf.piece_size();
  is_train_ = conf.is_train();
  floating_point_type_ = conf.floating_point_type();
}

void JobDesc::InitFromProto(const JobDescProto& proto) {
  train_dlnet_conf_ = proto.train_dlnet_conf();
  resource_ = proto.resource();
  strategy_ = proto.strategy();
  md_load_machine_ = proto.model_load_machine();
  md_save_machine_ = proto.model_save_machine();
  batch_size_ = proto.batch_size();
  piece_size_ = proto.piece_size();
  is_train_ = proto.is_train();
  floating_point_type_ = proto.floating_point_type();
}

void JobDesc::ToProto(JobDescProto* proto) const {
  *(proto->mutable_train_dlnet_conf()) = train_dlnet_conf_;
  *(proto->mutable_resource()) = resource_;
  *(proto->mutable_strategy()) = strategy_;
  *(proto->mutable_model_load_machine()) = md_load_machine_;
  *(proto->mutable_model_save_machine()) = md_save_machine_;
  proto->set_batch_size(batch_size_);
  proto->set_piece_size(piece_size_);
  proto->set_is_train(is_train_);
  proto->set_floating_point_type(floating_point_type_);
}

} // namespace oneflow
