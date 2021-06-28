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

#include "oneflow/user/data/dataset.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/common/str_util.h"
#include "oneflow/core/framework/op_kernel.h"
#include "oneflow/core/persistence/persistent_in_stream.h"
#include "oneflow/core/job/job_set.pb.h"

namespace oneflow {
namespace data {

class OFRecordDataset final : public Dataset<TensorBuffer> {
 public:
  using LoadTargetPtr = std::shared_ptr<TensorBuffer>;
  using LoadTargetPtrList = std::vector<LoadTargetPtr>;
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
      data_file_paths_.push_back(
          JoinPath(data_dir, part_name_prefix + std::string(zero_count, '0') + num));
    }

    parallel_id_ = ctx->parallel_ctx().parallel_id();
    parallel_num_ = ctx->parallel_ctx().parallel_num();
    CHECK_LE(parallel_num_, data_part_num_);
    BalancedSplitter bs(data_part_num_, parallel_num_);
    range_ = bs.At(parallel_id_);
    std::vector<std::string> local_file_paths = GetLocalFilePaths();
    save_to_local_ = Global<const IOConf>::Get()->save_downloaded_file_to_local_fs();
    in_stream_.reset(
        new PersistentInStream(DataFS(), local_file_paths, !shuffle_after_epoch_, save_to_local_));
  }
  ~OFRecordDataset() = default;

  LoadTargetPtrList Next() override {
    LoadTargetPtrList ret;
    LoadTargetPtr sample_ptr(new TensorBuffer());
    ReadSample(*sample_ptr);
    ret.push_back(std::move(sample_ptr));
    return ret;
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
    in_stream_.reset(new PersistentInStream(DataFS(), local_file_paths, false, save_to_local_));
  }

  std::vector<std::string> GetLocalFilePaths() {
    std::vector<std::string> ret;
    for (int i = range_.begin(); i < range_.end(); ++i) { ret.push_back(data_file_paths_.at(i)); }
    return ret;
  }

  int32_t current_epoch_;
  bool shuffle_after_epoch_;

  int32_t data_part_num_;
  int32_t parallel_id_;
  int32_t parallel_num_;
  Range range_;
  std::vector<std::string> data_file_paths_;
  bool save_to_local_;
  std::unique_ptr<PersistentInStream> in_stream_;
};

}  // namespace data
}  // namespace oneflow

#endif  // ONEFLOW_USER_DATA_OFRECORD_DATASET_H_
