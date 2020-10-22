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
#ifndef ONEFLOW_CUSTOMIZED_DATA_ONEREC_DATASET_H_
#define ONEFLOW_CUSTOMIZED_DATA_ONEREC_DATASET_H_

#include "oneflow/core/common/blocking_counter.h"
#include "oneflow/user/data/dataset.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/common/str_util.h"
#include "oneflow/core/framework/op_kernel.h"
#include "oneflow/core/persistence/persistent_in_stream.h"
#include "oneflow/core/job/job_set.pb.h"

#define XXH_NAMESPACE LZ4_
#include <xxhash.h>

namespace oneflow {

namespace {

constexpr int64_t kMaxPayloadSize = std::numeric_limits<int32_t>::max();
constexpr int64_t kMagicNumber = 0x24434552454E4F5E;  // '^ONEREC$', little endian
constexpr int32_t kReservedNumber = 0;
constexpr int32_t kPayloadAlignmentSize = 8;
constexpr int32_t kMagicFieldSize = 8;
constexpr int32_t kReservedFieldSize = 4;
constexpr int32_t kPayloadSizeFieldSize = 4;
constexpr int32_t kDigestFieldSize = 8;
constexpr int32_t kHeaderSizeWithoutDigest =
    kMagicFieldSize + kReservedFieldSize + kPayloadSizeFieldSize;
constexpr int32_t kHeaderSize = kHeaderSizeWithoutDigest + kDigestFieldSize;

inline XXH64_hash_t ByteSwap(XXH64_hash_t x) {
  return ((x & 0xff00000000000000ull) >> 56u) | ((x & 0x00ff000000000000ull) >> 40u)
         | ((x & 0x0000ff0000000000ull) >> 24u) | ((x & 0x000000ff00000000ull) >> 8u)
         | ((x & 0x00000000ff000000ull) << 8u) | ((x & 0x0000000000ff0000ull) << 24u)
         | ((x & 0x000000000000ff00ull) << 40u) | ((x & 0x00000000000000ffull) << 56u);
}

struct OneRecFrameHeader {
  int64_t magic;
  int32_t reserved;
  int32_t payload_size;
  XXH64_hash_t digest;
};

union OneRecFrameHeaderView {
  char raw[kHeaderSize];
  OneRecFrameHeader header;
};

union OneRecFrameFooterView {
  char raw[kDigestFieldSize];
  XXH64_hash_t digest;
};

}  // namespace

namespace data {

class OneRecDataset final : public Dataset<TensorBuffer> {
 public:
  using LoadTargetPtr = std::shared_ptr<TensorBuffer>;
  using LoadTargetPtrList = std::vector<LoadTargetPtr>;
  OF_DISALLOW_COPY_AND_MOVE(OneRecDataset);
  OneRecDataset(user_op::KernelInitContext* ctx, int32_t batch_size) : batch_size_(batch_size) {
    current_epoch_ = 0;
    shuffle_after_epoch_ = ctx->Attr<bool>("shuffle_after_epoch");
    data_file_paths_ = ctx->Attr<std::vector<std::string>>("files");
    parallel_id_ = ctx->parallel_ctx().parallel_id();
    parallel_num_ = ctx->parallel_ctx().parallel_num();
    BalancedSplitter bs(data_file_paths_.size(), parallel_num_);
    range_ = bs.At(parallel_id_);
    ResetInstream();
    hash_state_ = LZ4_XXH64_createState();
  }

  ~OneRecDataset() { CHECK_NE(LZ4_XXH64_freeState(hash_state_), XXH_ERROR); }

  LoadTargetPtrList Next() override {
    LoadTargetPtrList ret;
    ret.resize(batch_size_);
    for (int32_t i = 0; i < batch_size_; ++i) {
      ret.at(i).reset(new TensorBuffer());
      ReadSample(*ret.at(i).get());
    }
    return ret;
  }

 private:
  void ReadSample(TensorBuffer& tensor) {
    static_assert(sizeof(OneRecFrameHeader) == kHeaderSize, "");
    OneRecFrameHeaderView header_view{};
    static_assert(sizeof(header_view.header) == kHeaderSize, "");
    int32_t read_status = in_stream_->ReadFully(header_view.raw, kHeaderSize);
    if (read_status == -1) {
      ResetInstream();
      current_epoch_++;
      CHECK_EQ(in_stream_->ReadFully(header_view.raw, kHeaderSize), 0);
    } else {
      CHECK_EQ(read_status, 0);
    }
    CHECK_EQ(header_view.header.magic, kMagicNumber);
    CHECK_EQ(header_view.header.reserved, kReservedNumber);
    const int32_t payload_size = header_view.header.payload_size;
    CHECK_GE(payload_size, 0);
    CHECK_LE(payload_size, kMaxPayloadSize);
    XXH64_hash_t const seed = 0;
    CHECK_NE(LZ4_XXH64_reset(hash_state_, seed), XXH_ERROR);
    CHECK_NE(XXH64_update(hash_state_, header_view.raw, kHeaderSizeWithoutDigest), XXH_ERROR);
    CHECK_EQ(ByteSwap(header_view.header.digest), LZ4_XXH64_digest(hash_state_));
    const int32_t padded_size = RoundUp(payload_size, kPayloadAlignmentSize) - payload_size;
    tensor.Resize(Shape({payload_size}), DataType::kChar);
    char* body = tensor.mut_data<char>();
    CHECK_EQ(in_stream_->ReadFully(body, payload_size), 0);
    char padded[kPayloadAlignmentSize];
    CHECK_EQ(in_stream_->ReadFully(padded, padded_size), 0);  // read padded
    static_assert(sizeof(OneRecFrameFooterView) == kDigestFieldSize, "");
    OneRecFrameFooterView footer_view{};
    CHECK_EQ(in_stream_->ReadFully(footer_view.raw, kDigestFieldSize), 0);  // read footer
    CHECK_NE(XXH64_reset(hash_state_, seed), XXH_ERROR);
    CHECK_NE(LZ4_XXH64_update(hash_state_, body, payload_size), XXH_ERROR);
    CHECK_EQ(ByteSwap(footer_view.digest), LZ4_XXH64_digest(hash_state_));
  }

  void ResetInstream() {
    if (shuffle_after_epoch_) {
      std::mt19937 g(kOneflowDatasetSeed + current_epoch_);
      std::shuffle(data_file_paths_.begin(), data_file_paths_.end(), g);
    }
    std::vector<std::string> file_paths = GetLocalFilePaths();
    in_stream_.reset(new PersistentInStream(DataFS(), file_paths, false, false));
  }

  std::vector<std::string> GetLocalFilePaths() {
    std::vector<std::string> ret;
    for (int i = range_.begin(); i < range_.end(); ++i) { ret.push_back(data_file_paths_.at(i)); }
    return ret;
  }

  int32_t current_epoch_;
  bool shuffle_after_epoch_;

  int32_t parallel_id_;
  int32_t parallel_num_;
  Range range_;
  std::vector<std::string> data_file_paths_;
  std::unique_ptr<PersistentInStream> in_stream_;
  XXH64_state_t* hash_state_;
  int32_t batch_size_;
};

}  // namespace data
}  // namespace oneflow

#endif  // ONEFLOW_CUSTOMIZED_DATA_ONEREC_DATASET_H_
