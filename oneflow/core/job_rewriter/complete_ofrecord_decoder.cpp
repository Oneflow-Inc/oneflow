/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include "oneflow/core/job_rewriter/job_pass.h"
#include "oneflow/core/job/job_builder.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/job/parallel_desc.h"

namespace oneflow {

namespace {

void SplitDecodeOps(Job* job) {
  std::vector<OperatorConf> gen_op_confs;
  for (OperatorConf& op_conf : *(job->mutable_net()->mutable_op())) {
    if (op_conf.has_decode_ofrecord_conf() == false) { continue; }
    if (op_conf.decode_ofrecord_conf().blob_size() == 1) { continue; }
    const DecodeOFRecordOpConf& decode_conf = op_conf.decode_ofrecord_conf();
    PbRpf<BlobConf>* blobs = op_conf.mutable_decode_ofrecord_conf()->mutable_blob();
    Erase<PbRpf<BlobConf>>(
        *blobs,
        [&](const BlobConf& blob_conf) -> bool { return blob_conf.max_sequence_size() > 1; },
        [&](const BlobConf& blob_conf) {
          gen_op_confs.emplace_back(op_conf);
          DecodeOFRecordOpConf* gen_decode_conf =
              gen_op_confs.back().mutable_decode_ofrecord_conf();
          *gen_decode_conf = decode_conf;
          gen_decode_conf->clear_blob();
          *gen_decode_conf->add_blob() = blob_conf;
        });
  }
  // TODO: erase decode op which has no blob any more
  for (OperatorConf& gen_op_conf : gen_op_confs) { *(job->mutable_net()->add_op()) = gen_op_conf; }
}

void AddRecordLoadOps(Job* job) {
  HashMap<std::pair<std::string, std::string>, std::vector<OperatorConf*>> data_info2decode_ops;
  HashMap<std::pair<std::string, std::string>, int32_t> data_info2part_num;
  HashMap<std::pair<std::string, std::string>, int32_t> data_info2suffix_length;
  HashMap<std::pair<std::string, std::string>, const RandomShuffleConf*> data_info2shuffle_conf;
  size_t op_num = job->net().op_size();
  FOR_RANGE(size_t, idx, 0, op_num) {
    OperatorConf* op_conf = job->mutable_net()->mutable_op()->Mutable(idx);
    if (op_conf->has_decode_ofrecord_conf() == false) { continue; }
    const DecodeOFRecordOpConf& decode_conf = op_conf->decode_ofrecord_conf();
    if (decode_conf.has_in()) { continue; }
    if (decode_conf.blob_size() == 0) { continue; }
    std::pair<std::string, std::string> data_info = {decode_conf.data_dir(),
                                                     decode_conf.part_name_prefix()};
    data_info2decode_ops[data_info].emplace_back(op_conf);
    int32_t part_name_suffix_length = decode_conf.part_name_suffix_length();
    int32_t data_part_num = decode_conf.data_part_num();
    if (data_info2suffix_length.find(data_info) != data_info2suffix_length.end()) {
      CHECK_EQ(data_info2suffix_length[data_info], part_name_suffix_length);
      CHECK_EQ(data_info2part_num[data_info], data_part_num);
    } else {
      data_info2suffix_length[data_info] = part_name_suffix_length;
      data_info2part_num[data_info] = data_part_num;
    }
    const RandomShuffleConf* shuffle_conf =
        decode_conf.has_random_shuffle_conf() ? &decode_conf.random_shuffle_conf() : nullptr;
    if (data_info2shuffle_conf.find(data_info) != data_info2shuffle_conf.end()) {
      if (shuffle_conf == nullptr) {
        CHECK(data_info2shuffle_conf.at(data_info) == nullptr);
      } else {
        CHECK(data_info2shuffle_conf.at(data_info) != nullptr);
        CHECK_EQ(data_info2shuffle_conf.at(data_info)->buffer_size(), shuffle_conf->buffer_size());
      }
    } else {
      CHECK(data_info2shuffle_conf.emplace(data_info, shuffle_conf).second);
    }
  }

  HashMap<std::string, const ParallelConf*> name2parallel_conf;
  for (const PlacementGroup& p_group : job->placement().placement_group()) {
    for (const std::string& op_name : p_group.op_set().op_name()) {
      CHECK(name2parallel_conf.emplace(op_name, &p_group.parallel_conf()).second);
    }
  }

  for (const auto& pair : data_info2decode_ops) {
    HashMap<int64_t, std::vector<OperatorConf*>> batch_size2decode_ops;
    for (const auto& decode_op : pair.second) {
      batch_size2decode_ops[decode_op->decode_ofrecord_conf().batch_size()].push_back(decode_op);
    }
    for (const auto& batch_size_n_decode_ops : batch_size2decode_ops) {
      std::vector<const ParallelConf*> parallel_confs;
      for (const OperatorConf* op_conf : batch_size_n_decode_ops.second) {
        auto op_parallel_conf_it = name2parallel_conf.find(op_conf->name());
        CHECK(op_parallel_conf_it != name2parallel_conf.end());
        auto iter = std::find_if(
            parallel_confs.begin(), parallel_confs.end(), [&](const ParallelConf* parallel_conf) {
              PbMd message_diff;
              return message_diff.Equivalent(*parallel_conf, *(op_parallel_conf_it->second));
            });
        if (iter == parallel_confs.end()) {
          parallel_confs.emplace_back(op_parallel_conf_it->second);
        }
      }
      LOG_IF(WARNING, parallel_confs.size() > 1)
          << "Operators sharing the same data information belong to different placement groups";
      for (const ParallelConf* parallel_conf : parallel_confs) {
        std::string record_load_op_name = "loader" + NewUniqueId();
        std::string record_load_out_name = "out";
        std::string record_load_lbi_name = record_load_op_name + "/" + record_load_out_name;
        OperatorConf* op = job->mutable_net()->add_op();
        RecordLoadOpConf* record_load_op = op->mutable_record_load_conf();
        op->set_name(record_load_op_name);
        record_load_op->set_out(record_load_out_name);
        record_load_op->set_data_dir(pair.first.first);
        record_load_op->set_data_part_num(data_info2part_num.at(pair.first));
        record_load_op->set_part_name_prefix(pair.first.second);
        record_load_op->set_part_name_suffix_length(data_info2suffix_length.at(pair.first));
        record_load_op->set_batch_size(batch_size_n_decode_ops.first);
        if (data_info2shuffle_conf.at(pair.first) != nullptr) {
          *record_load_op->mutable_random_shuffle_conf() = *data_info2shuffle_conf.at(pair.first);
        }
        PlacementGroup* p_group = job->mutable_placement()->add_placement_group();
        *(p_group->mutable_op_set()->add_op_name()) = record_load_op_name;
        *(p_group->mutable_parallel_conf()) = *parallel_conf;
        for (OperatorConf* op : batch_size_n_decode_ops.second) {
          std::string op_name = op->name();
          auto op_parallel_conf_it = name2parallel_conf.find(op_name);
          CHECK(op_parallel_conf_it != name2parallel_conf.end());
          PbMd message_diff;
          if (!message_diff.Equivalent(*parallel_conf, *(op_parallel_conf_it->second))) {
            continue;
          }
          op->mutable_decode_ofrecord_conf()->set_in(record_load_lbi_name);
        }
      }
    }
  }
}

}  // namespace

class CompleteOfrecordDecoder final : public JobPass {
 public:
  Maybe<void> Apply(Job* job, JobPassCtx* ctx) const override { return Apply(job); }

  Maybe<void> Apply(Job* job) const {
    SplitDecodeOps(job);
    AddRecordLoadOps(job);
    return Maybe<void>::Ok();
  }
};

REGISTER_JOB_PASS("CompleteOfrecordDecoder", CompleteOfrecordDecoder);

}  // namespace oneflow
