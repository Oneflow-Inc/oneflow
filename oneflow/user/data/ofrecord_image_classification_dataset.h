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
#ifndef ONEFLOW_USER_DATA_OFRECORD_IMAGE_CLASSIFICATION_DATASET_H_
#define ONEFLOW_USER_DATA_OFRECORD_IMAGE_CLASSIFICATION_DATASET_H_

#include "oneflow/core/thread/thread_pool.h"
#include "oneflow/core/common/buffer.h"
#include "oneflow/user/data/dataset.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/common/str_util.h"
#include "oneflow/core/framework/op_kernel.h"
#include "oneflow/core/persistence/persistent_in_stream.h"
#include "oneflow/core/job/job_set.pb.h"
#include "oneflow/user/data/ofrecord_dataset.h"
#include "oneflow/user/image/image_util.h"
#include "oneflow/core/job/resource_desc.h"
#include "oneflow/core/job/global_for.h"
#include <opencv2/opencv.hpp>

namespace oneflow {

namespace data {

struct ImageClassificationDataInstance {
  std::shared_ptr<TensorBuffer> label;
  std::shared_ptr<TensorBuffer> image;
};

using BaseDataset = Dataset<TensorBuffer>;
using BaseLoadTargetPtr = BaseDataset::LoadTargetPtr;
using BaseLoadTargetPtrList = BaseDataset::LoadTargetPtrList;

namespace {

void DecodeImageFromOFRecord(const OFRecord& record, const std::string& feature_name,
                             const std::string& color_space, TensorBuffer* out) {
  auto image_feature_it = record.feature().find(feature_name);
  CHECK(image_feature_it != record.feature().end());
  const Feature& image_feature = image_feature_it->second;
  CHECK(image_feature.has_bytes_list());
  CHECK(image_feature.bytes_list().value_size() == 1);
  const std::string& src_data = image_feature.bytes_list().value(0);
  cv::Mat image = cv::imdecode(cv::Mat(1, src_data.size(), CV_8UC1, (void*)(src_data.data())),
                               cv::IMREAD_COLOR);
  int W = image.cols;
  int H = image.rows;

  // convert color space
  if (ImageUtil::IsColor(color_space) && color_space != "BGR") {
    ImageUtil::ConvertColor("BGR", image, color_space, image);
  }

  CHECK(image.isContinuous());
  const int c = ImageUtil::IsColor(color_space) ? 3 : 1;
  CHECK_EQ(c, image.channels());
  Shape image_shape({H, W, c});
  out->Resize(image_shape, DataType::kUInt8);
  CHECK_EQ(image_shape.elem_cnt(), out->nbytes());
  CHECK_EQ(image_shape.elem_cnt(), image.total() * image.elemSize());
  memcpy(out->mut_data<uint8_t>(), image.ptr(), image_shape.elem_cnt());
}

void DecodeLabelFromFromOFRecord(const OFRecord& record, const std::string& feature_name,
                                 TensorBuffer* out) {
  auto label_feature_it = record.feature().find(feature_name);
  CHECK(label_feature_it != record.feature().end());
  const Feature& label_feature = label_feature_it->second;
  out->Resize(Shape({1}), DataType::kInt32);
  if (label_feature.has_int32_list()) {
    CHECK_EQ(label_feature.int32_list().value_size(), 1);
    *out->mut_data<int32_t>() = label_feature.int32_list().value(0);
  } else if (label_feature.has_int64_list()) {
    CHECK_EQ(label_feature.int64_list().value_size(), 1);
    *out->mut_data<int32_t>() = label_feature.int64_list().value(0);
  } else {
    UNIMPLEMENTED();
  }
}

void LoadWorker(BaseDataset* record_dataset,
                std::vector<std::unique_ptr<Buffer<BaseLoadTargetPtr>>>* decode_in_buffers) {
  int64_t thread_idx = 0;
  bool shutdown = false;
  while (!shutdown) {
    BaseLoadTargetPtrList records = record_dataset->Next();
    for (const auto& record : records) {
      auto& current_in_buffer = decode_in_buffers->at(thread_idx);
      thread_idx = (thread_idx + 1) % decode_in_buffers->size();
      auto status = current_in_buffer->Send(record);
      if (status == kBufferStatusErrorClosed) {
        shutdown = true;
        break;
      }
      CHECK(status == kBufferStatusSuccess);
    }
  }
}

void DecodeWorker(const std::string image_feature_name, const std::string label_feature_name,
                  const std::string color_space, Buffer<BaseLoadTargetPtr>* in_buffer,
                  Buffer<std::shared_ptr<ImageClassificationDataInstance>>* out_buffer) {
  while (true) {
    BaseLoadTargetPtr serialized_record;
    auto receive_status = in_buffer->Receive(&serialized_record);
    if (receive_status == kBufferStatusErrorClosed) { break; }
    CHECK(receive_status == kBufferStatusSuccess);
    OFRecord record;
    CHECK(record.ParseFromArray(serialized_record->data<char>(),
                                serialized_record->shape().elem_cnt()));
    std::shared_ptr<ImageClassificationDataInstance> instance(
        new ImageClassificationDataInstance());
    instance->image.reset(new TensorBuffer());
    DecodeImageFromOFRecord(record, image_feature_name, color_space, instance->image.get());
    instance->label.reset(new TensorBuffer());
    DecodeLabelFromFromOFRecord(record, label_feature_name, instance->label.get());
    auto send_status = out_buffer->Send(instance);
    if (send_status == kBufferStatusErrorClosed) { break; }
    CHECK(send_status == kBufferStatusSuccess);
  }
}

int32_t GetNumLocalDecodeThreads(int32_t num_decode_threads_per_machine,
                                 const ParallelDesc& parallel_desc,
                                 const ParallelContext& parallel_ctx) {
  if (num_decode_threads_per_machine == 0) {
    num_decode_threads_per_machine =
        Global<ResourceDesc, ForSession>::Get()->ComputeThreadPoolSize();
  }
  int64_t machine_id = CHECK_JUST(parallel_desc.MachineId4ParallelId(parallel_ctx.parallel_id()));
  int64_t parallel_num_on_this_machine = parallel_desc.sorted_dev_phy_ids(machine_id).size();
  return std::max<int32_t>(num_decode_threads_per_machine / parallel_num_on_this_machine, 1);
}

}  // namespace

class OFRecordImageClassificationDataset final : public Dataset<ImageClassificationDataInstance> {
 public:
  using LoadTargetPtr = std::shared_ptr<ImageClassificationDataInstance>;
  using LoadTargetPtrList = std::vector<LoadTargetPtr>;
  OF_DISALLOW_COPY_AND_MOVE(OFRecordImageClassificationDataset);
  OFRecordImageClassificationDataset(user_op::KernelInitContext* ctx,
                                     std::unique_ptr<BaseDataset>&& base)
      : base_(std::move(base)), out_thread_idx_(0) {
    const std::string& color_space = ctx->Attr<std::string>("color_space");
    const std::string& image_feature_name = ctx->Attr<std::string>("image_feature_name");
    const std::string& label_feature_name = ctx->Attr<std::string>("label_feature_name");
    const auto num_decode_threads_per_machine =
        ctx->Attr<int32_t>("num_decode_threads_per_machine");
    const auto decode_buffer_size_per_thread = ctx->Attr<int32_t>("decode_buffer_size_per_thread");
    const int32_t num_local_decode_threads = GetNumLocalDecodeThreads(
        num_decode_threads_per_machine, ctx->parallel_desc(), ctx->parallel_ctx());
    decode_in_buffers_.resize(num_local_decode_threads);
    decode_out_buffers_.resize(num_local_decode_threads);
    for (int64_t i = 0; i < num_local_decode_threads; ++i) {
      decode_in_buffers_.at(i).reset(new Buffer<BaseLoadTargetPtr>(decode_buffer_size_per_thread));
      decode_out_buffers_.at(i).reset(new Buffer<LoadTargetPtr>(decode_buffer_size_per_thread));
      decode_threads_.emplace_back(
          std::thread(&DecodeWorker, image_feature_name, label_feature_name, color_space,
                      decode_in_buffers_.at(i).get(), decode_out_buffers_.at(i).get()));
    }
    load_thread_ = std::thread(&LoadWorker, base_.get(), &decode_in_buffers_);
  }
  ~OFRecordImageClassificationDataset() override {
    for (auto& out_buffer : decode_out_buffers_) { out_buffer->Close(); }
    for (auto& in_buffer : decode_in_buffers_) { in_buffer->Close(); }
    load_thread_.join();
    for (auto& decode_thread : decode_threads_) { decode_thread.join(); }
  }

  LoadTargetPtrList Next() override {
    LoadTargetPtrList ret;
    LoadTargetPtr sample_ptr;
    size_t thread_idx =
        out_thread_idx_.fetch_add(1, std::memory_order_relaxed) % decode_out_buffers_.size();
    auto status = decode_out_buffers_.at(thread_idx)->Receive(&sample_ptr);
    CHECK_EQ(status, kBufferStatusSuccess);
    ret.push_back(std::move(sample_ptr));
    return ret;
  }

 private:
  std::unique_ptr<BaseDataset> base_;
  std::thread load_thread_;
  std::vector<std::thread> decode_threads_;
  std::vector<std::unique_ptr<Buffer<BaseLoadTargetPtr>>> decode_in_buffers_;
  std::vector<std::unique_ptr<Buffer<LoadTargetPtr>>> decode_out_buffers_;
  std::atomic<size_t> out_thread_idx_;
};

}  // namespace data

}  // namespace oneflow

#endif  // ONEFLOW_USER_DATA_OFRECORD_IMAGE_CLASSIFICATION_DATASET_H_
