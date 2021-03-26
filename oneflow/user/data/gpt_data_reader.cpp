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
#include "oneflow/user/data/gpt_data_reader.h"
#include "oneflow/user/data/gpt_dataset.h"
// #include "oneflow/user/data/distributed_training_dataset.h"
// #include "oneflow/user/data/group_batch_dataset.h"
// #include "oneflow/user/data/batch_dataset.h"
// #include "oneflow/core/persistence/file_system.h"
// #include "oneflow/core/persistence/persistent_in_stream.h"

namespace oneflow {

namespace data {

// COCODataReader::COCODataReader(user_op::KernelInitContext* ctx) : DataReader<COCOImage>(ctx) {
//   std::shared_ptr<const COCOMeta> meta(new COCOMeta(
//       ctx->Attr<int64_t>("session_id"), ctx->Attr<std::string>("annotation_file"),
//       ctx->Attr<std::string>("image_dir"),
//       ctx->Attr<bool>("remove_images_without_annotations")));

//   std::unique_ptr<RandomAccessDataset<COCOImage>> coco_dataset_ptr(new COCODataset(ctx, meta));
//   loader_.reset(new DistributedTrainingDataset<COCOImage>(
//       ctx->parallel_ctx().parallel_num(), ctx->parallel_ctx().parallel_id(),
//       ctx->Attr<bool>("stride_partition"), ctx->Attr<bool>("shuffle_after_epoch"),
//       ctx->Attr<int64_t>("random_seed"), std::move(coco_dataset_ptr)));

//   size_t batch_size = ctx->TensorDesc4ArgNameAndIndex("image", 0)->shape().elem_cnt();
//   if (ctx->Attr<bool>("group_by_ratio")) {
//     auto GetGroupId = [](const std::shared_ptr<COCOImage>& sample) {
//       return static_cast<int64_t>(sample->height / sample->width);
//     };
//     loader_.reset(new GroupBatchDataset<COCOImage>(batch_size, GetGroupId, std::move(loader_)));
//   } else {
//     loader_.reset(new BatchDataset<COCOImage>(batch_size, std::move(loader_)));
//   }

//   parser_.reset(new COCOParser(meta));
//   StartLoadThread();
// }

}  // namespace data

}  // namespace oneflow
