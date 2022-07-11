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
#ifndef ONEFLOW_USER_DATA_OFRECORD_DATASET_H_
#define ONEFLOW_USER_DATA_OFRECORD_DATASET_H_

#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/common/str_util.h"
#include "oneflow/core/framework/op_kernel.h"
#include "oneflow/core/persistence/persistent_in_stream.h"
#include "oneflow/core/job/job_set.pb.h"
#include "oneflow/core/rpc/include/global_process_ctx.h"
#include "oneflow/core/job/env_desc.h"
#include "oneflow/user/data/dataset.h"

namespace oneflow {
namespace data {

class OFRecordDataset final : public Dataset<TensorBuffer> {
 public:
  using Base = Dataset<TensorBuffer>;
  using SampleType = typename Base::SampleType;
  using BatchType = typename Base::BatchType;

  OF_DISALLOW_COPY_AND_MOVE(OFRecordDataset);

  OFRecordDataset(user_op::KernelInitContext* ctx) {
    current_epoch_ = 0;
    shuffle_after_epoch_ = ctx->Attr<bool>("shuffle_after_epoch");

    // in stream
    data_part_num_ = ctx->Attr<int32_t>("data_part_num");
    std::string data_dir = ctx->Attr<std::string>("data_dir");
    std::string part_name_prefix = ctx->Attr<std::string>("part_name_prefix");
    int32_t part_name_suffix_length = ctx->Attr<int32_t>("part_name_suffix_length");

    for (int i = 0; i < data_part_num_; ++i) {
      std::string num = std::to_string(i);
      int32_t zero_count =
          std::max(part_name_suffix_length - static_cast<int32_t>(num.length()), 0);
      data_file_paths_.emplace_back(
          JoinPath(data_dir, part_name_prefix + std::string(zero_count, '0') + num));
    }

    bool is_local = false;
    // NOTE(zwx): OFRecordDataset is used by OFRecordDataReader and
    // OFRecordImageClassificationDataReader both, the latter has no attr nd_sbp,
    // so it couldn't work in DDP for now. The If condition here could be removed when
    // OFRecordImageClassificationDataReader had supported DDP (add attr nd_sbp)
    // or been deprecated.
    if (ctx->op_type_name() == "OFRecordReader") {
      auto nd_sbp_str_vec = ctx->Attr<std::vector<std::string>>("nd_sbp");
      // NOTE(zwx): OFRecordDataset is not global since attr nd_sbp is empty,
      // we assume that it works in DDP
      if (nd_sbp_str_vec.empty()) { is_local = true; }
    }
    if (is_local) {
      parallel_id_ = GlobalProcessCtx::Rank();
      parallel_num_ = GlobalProcessCtx::WorldSize();
    } else {
      parallel_id_ = ctx->parallel_ctx().parallel_id();
      parallel_num_ = ctx->parallel_ctx().parallel_num();
    }
    CHECK_LE(parallel_num_, data_part_num_);
    BalancedSplitter bs(data_part_num_, parallel_num_);
    range_ = bs.At(parallel_id_);
    std::vector<std::string> local_file_paths = GetLocalFilePaths();
    in_stream_.reset(
        new PersistentInStream(DataFS(), local_file_paths, !shuffle_after_epoch_, false));
  }
  ~OFRecordDataset() = default;

  BatchType Next() override {
    BatchType batch;
    batch.push_back(TensorBuffer());
    ReadSample(batch.back());
    return batch;
  }

 private:
  void ReadSample(TensorBuffer& tensor) {
    int64_t OFRecord_size = -1;
    char* size_ptr = reinterpret_cast<char*>(&OFRecord_size);
    if (in_stream_->ReadFully(size_ptr, sizeof(int64_t)) != 0) {
      ShuffleAfterEpoch();
      CHECK_EQ(in_stream_->ReadFully(size_ptr, sizeof(int64_t)), 0);
    }
    CHECK_GT(OFRecord_size, 0);
    tensor.Resize(Shape({OFRecord_size}), DataType::kChar);
    CHECK_EQ(in_stream_->ReadFully(tensor.mut_data<char>(), OFRecord_size), 0);
  }

  void ShuffleAfterEpoch() {
    CHECK(shuffle_after_epoch_);
    current_epoch_++;  // move to next epoch
    std::mt19937 g(kOneflowDatasetSeed + current_epoch_);
    std::shuffle(data_file_paths_.begin(), data_file_paths_.end(), g);
    std::vector<std::string> local_file_paths = GetLocalFilePaths();
    in_stream_.reset(new PersistentInStream(DataFS(), local_file_paths, false, false));
  }

  std::vector<std::string> GetLocalFilePaths() {
    std::vector<std::string> ret;
    for (int i = range_.begin(); i < range_.end(); ++i) {
      ret.emplace_back(data_file_paths_.at(i));
    }
    return ret;
  }

  int32_t current_epoch_;
  bool shuffle_after_epoch_;

  int32_t data_part_num_;
  int32_t parallel_id_;
  int32_t parallel_num_;
  Range range_;
  std::vector<std::string> data_file_paths_;
  std::unique_ptr<PersistentInStream> in_stream_;
};

}  // namespace data
}  // namespace oneflow

#endif  // ONEFLOW_USER_DATA_OFRECORD_DATASET_H_
