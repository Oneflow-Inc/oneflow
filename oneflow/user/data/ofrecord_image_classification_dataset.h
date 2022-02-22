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

#include "oneflow/user/data/dataset.h"
#include "oneflow/core/common/buffer.h"
#include "oneflow/core/framework/op_kernel.h"

namespace oneflow {

namespace data {

struct ImageClassificationDataInstance {
  TensorBuffer label;
  TensorBuffer image;
};

class OFRecordImageClassificationDataset final : public Dataset<ImageClassificationDataInstance> {
 public:
  using Base = Dataset<ImageClassificationDataInstance>;
  using SampleType = Base::SampleType;
  using BatchType = Base::BatchType;
  using NestedDS = Dataset<TensorBuffer>;
  using NestedSampleType = NestedDS::SampleType;

  OF_DISALLOW_COPY_AND_MOVE(OFRecordImageClassificationDataset);

  OFRecordImageClassificationDataset(user_op::KernelInitContext* ctx,
                                     std::unique_ptr<NestedDS>&& dataset);
  ~OFRecordImageClassificationDataset() override;

  BatchType Next() override {
    size_t thread_idx =
        out_thread_idx_.fetch_add(1, std::memory_order_relaxed) % decode_out_buffers_.size();
    CHECK_LT(thread_idx, decode_out_buffers_.size());

    BatchType batch;
    SampleType sample;
    auto status = decode_out_buffers_[thread_idx]->Pull(&sample);
    CHECK_EQ(status, kBufferStatusSuccess);
    batch.push_back(std::move(sample));
    return batch;
  }

 private:
  std::unique_ptr<NestedDS> nested_ds_;
  std::thread load_thread_;
  std::vector<std::thread> decode_threads_;
  std::vector<std::unique_ptr<Buffer<NestedSampleType>>> decode_in_buffers_;
  std::vector<std::unique_ptr<Buffer<SampleType>>> decode_out_buffers_;
  std::atomic<size_t> out_thread_idx_;
};

}  // namespace data

}  // namespace oneflow

#endif  // ONEFLOW_USER_DATA_OFRECORD_IMAGE_CLASSIFICATION_DATASET_H_
