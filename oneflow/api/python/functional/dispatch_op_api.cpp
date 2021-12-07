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

#include "oneflow/core/common/scalar.h"
#include "oneflow/core/framework/nd_sbp.h"
#include "oneflow/core/framework/op_interp_ctx.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/framework/op_interp_ctx_generated.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/core/functional/functional.h"
#include "oneflow/core/functional/function_library.h"

namespace oneflow {
namespace one {
namespace functional {

namespace impl {

ONEFLOW_FUNCTION_LIBRARY(m) {
  m.add_functor(
      "DispatchFeedInput",
      [](const std::shared_ptr<OpExpr>& op, const std::shared_ptr<Tensor>& input) -> Maybe<Tensor> {
        auto ctx = std::make_shared<FeedInputOpInterpCtx>();
        return OpInterpUtil::Dispatch<Tensor>(*op, {input}, ctx);
      });
  m.add_functor(
      "DispatchFetchOutput",
      [](const std::shared_ptr<OpExpr>& op, const std::shared_ptr<Tensor>& input) -> Maybe<Tensor> {
        auto ctx = std::make_shared<FetchOutputOpInterpCtx>();
        return OpInterpUtil::Dispatch<Tensor>(*op, {input}, ctx);
      });
  m.add_functor("DispatchFeedVariable",
                [](const std::shared_ptr<OpExpr>& op, const std::shared_ptr<Tensor>& input,
                   const Scalar& l2) -> Maybe<Tensor> {
                  auto ctx = std::make_shared<FeedVariableOpInterpCtx>();
                  ctx->_l2 = JUST(l2.As<double>());
                  return OpInterpUtil::Dispatch<Tensor>(*op, {input}, ctx);
                });
  m.add_functor(
      "DispatchOfrecordReader",
      [](const std::shared_ptr<OpExpr>& op, const std::string& data_dir, int32_t data_part_num,
         const std::string& part_name_prefix, int32_t part_name_suffix_length, int32_t batch_size,
         int32_t shuffle_buffer_size, bool random_shuffle, bool shuffle_after_epoch, int64_t seed,
         const Optional<Symbol<Device>>& device) -> Maybe<Tensor> {
        auto ctx = std::make_shared<OFRecordReaderOpInterpCtxImpl<schema::OFRecordReaderOp>>();
        ctx->set_data_dir(data_dir);
        ctx->set_data_part_num(data_part_num);
        ctx->set_part_name_prefix(part_name_prefix);
        ctx->set_part_name_suffix_length(part_name_suffix_length);
        ctx->set_batch_size(batch_size);
        ctx->set_shuffle_buffer_size(shuffle_buffer_size);
        ctx->set_random_shuffle(random_shuffle);
        ctx->set_shuffle_after_epoch(shuffle_after_epoch);
        ctx->set_seed(seed);
        ctx->device = device;
        return OpInterpUtil::Dispatch<Tensor>(*op, {}, ctx);
      });
  m.add_functor(
      "DispatchOfrecordReader",
      [](const std::shared_ptr<OpExpr>& op, const std::string& data_dir, int32_t data_part_num,
         const std::string& part_name_prefix, int32_t part_name_suffix_length, int32_t batch_size,
         int32_t shuffle_buffer_size, bool random_shuffle, bool shuffle_after_epoch, int64_t seed,
         const Symbol<ParallelDesc>& placement,
         const std::vector<Symbol<cfg::SbpParallel>>& sbp_tuple) -> Maybe<Tensor> {
        auto ctx = std::make_shared<OFRecordReaderOpInterpCtxImpl<schema::OFRecordReaderOp>>();
        ctx->set_data_dir(data_dir);
        ctx->set_data_part_num(data_part_num);
        ctx->set_part_name_prefix(part_name_prefix);
        ctx->set_part_name_suffix_length(part_name_suffix_length);
        ctx->set_batch_size(batch_size);
        ctx->set_shuffle_buffer_size(shuffle_buffer_size);
        ctx->set_random_shuffle(random_shuffle);
        ctx->set_shuffle_after_epoch(shuffle_after_epoch);
        ctx->set_seed(seed);
        ctx->set_nd_sbp(*JUST(GetNdSbpStrList(sbp_tuple)));
        ctx->parallel_desc = placement;
        ctx->sbp = JUST(GetNdSbp(sbp_tuple));
        return OpInterpUtil::Dispatch<Tensor>(*op, {}, ctx);
      });
  m.add_functor(
      "DispatchOfrecordRawDecoder",
      [](const std::shared_ptr<OpExpr>& op, const std::shared_ptr<Tensor>& input,
         const std::string name, const Shape& shape, const Symbol<DType>& data_type,
         bool dim1_varying_length, bool truncate) -> Maybe<Tensor> {
        auto ctx =
            std::make_shared<OfrecordRawDecoderOpInterpCtxImpl<schema::OfrecordRawDecoderOp>>();
        ctx->set_name(name);
        ctx->set_shape(shape);
        ctx->set_data_type(data_type->data_type());
        ctx->set_dim1_varying_length(dim1_varying_length);
        ctx->set_truncate(truncate);
        return OpInterpUtil::Dispatch<Tensor>(*op, {input}, ctx);
      });
  m.add_functor(
      "DispatchCoinFlip",
      [](const std::shared_ptr<OpExpr>& op, int64_t batch_size, Scalar probability, int64_t seed,
         bool has_seed, const Optional<Symbol<Device>> device) -> Maybe<Tensor> {
        auto ctx = std::make_shared<CoinFlipOpInterpCtxImpl<schema::CoinFlipOp>>();
        ctx->set_probability(JUST(probability.As<float>()));
        ctx->set_batch_size(batch_size);
        ctx->set_seed(seed);
        ctx->set_has_seed(has_seed);
        ctx->device = device;
        return OpInterpUtil::Dispatch<Tensor>(*op, {}, ctx);
      });
  m.add_functor("DispatchCoinFlip",
                [](const std::shared_ptr<OpExpr>& op, int64_t batch_size, Scalar probability,
                   int64_t seed, bool has_seed, const Symbol<ParallelDesc>& placement,
                   const std::vector<Symbol<cfg::SbpParallel>>& sbp_tuple) -> Maybe<Tensor> {
                  auto ctx = std::make_shared<CoinFlipOpInterpCtxImpl<schema::CoinFlipOp>>();
                  ctx->set_probability(JUST(probability.As<float>()));
                  ctx->set_batch_size(batch_size);
                  ctx->set_seed(seed);
                  ctx->set_has_seed(has_seed);
                  ctx->set_nd_sbp(*JUST(GetNdSbpStrList(sbp_tuple)));
                  ctx->parallel_desc = placement;
                  ctx->sbp = JUST(GetNdSbp(sbp_tuple));
                  return OpInterpUtil::Dispatch<Tensor>(*op, {}, ctx);
                });
  m.add_functor(
      "DispatchCropMirrorNormalizeFromUint8",
      [](const std::shared_ptr<OpExpr>& op, const TensorTuple& input, int64_t crop_h,
         int64_t crop_w, float crop_pos_x, float crop_pos_y, const std::vector<float>& mean,
         const std::vector<float>& std, const Symbol<DType>& output_dtype,
         const std::string output_layout, const std::string color_space) -> Maybe<Tensor> {
        auto ctx = std::make_shared<
            CropMirrorNormalizeFromUint8OpInterpCtxImpl<schema::CropMirrorNormalizeFromUint8Op>>();
        ctx->set_color_space(color_space);
        ctx->set_output_layout(output_layout);
        ctx->set_mean(mean);
        ctx->set_std(std);
        ctx->set_crop_h(crop_h);
        ctx->set_crop_w(crop_w);
        ctx->set_crop_pos_x(crop_pos_x);
        ctx->set_crop_pos_y(crop_pos_y);
        ctx->set_output_dtype(output_dtype->data_type());
        return OpInterpUtil::Dispatch<Tensor>(*op, input, ctx);
      });
  m.add_functor(
      "DispatchCropMirrorNormalizeFromTensorBuffer",
      [](const std::shared_ptr<OpExpr>& op, const TensorTuple& input, int64_t crop_h,
         int64_t crop_w, float crop_pos_x, float crop_pos_y, const std::vector<float>& mean,
         const std::vector<float>& std, const Symbol<DType>& output_dtype,
         const std::string output_layout, const std::string color_space) -> Maybe<Tensor> {
        auto ctx = std::make_shared<CropMirrorNormalizeFromTensorbufferOpInterpCtxImpl<
            schema::CropMirrorNormalizeFromTensorbufferOp>>();
        ctx->set_color_space(color_space);
        ctx->set_output_layout(output_layout);
        ctx->set_mean(mean);
        ctx->set_std(std);
        ctx->set_crop_h(crop_h);
        ctx->set_crop_w(crop_w);
        ctx->set_crop_pos_x(crop_pos_x);
        ctx->set_crop_pos_y(crop_pos_y);
        ctx->set_output_dtype(output_dtype->data_type());
        return OpInterpUtil::Dispatch<Tensor>(*op, {input}, ctx);
      });
  m.add_functor(
      "DispatchOfrecordImageDecoderRandomCrop",
      [](const std::shared_ptr<OpExpr>& op, const std::shared_ptr<Tensor>& input,
         const std::string name, const std::string color_space,
         const std::vector<float>& random_area, const std::vector<float>& random_aspect_ratio,
         int32_t num_attempts, int64_t seed, bool has_seed) -> Maybe<Tensor> {
        auto ctx = std::make_shared<OfrecordImageDecoderRandomCropOpInterpCtxImpl<
            schema::OfrecordImageDecoderRandomCropOp>>();
        ctx->set_name(name);
        ctx->set_color_space(color_space);
        ctx->set_num_attempts(num_attempts);
        ctx->set_seed(seed);
        ctx->set_has_seed(has_seed);
        ctx->set_random_area(random_area);
        ctx->set_random_aspect_ratio(random_aspect_ratio);
        return OpInterpUtil::Dispatch<Tensor>(*op, {input}, ctx);
      });
  m.add_functor(
      "DispatchOfrecordImageDecoder",
      [](const std::shared_ptr<OpExpr>& op, const std::shared_ptr<Tensor>& input,
         const std::string name, const std::string color_space) -> Maybe<Tensor> {
        auto ctx =
            std::make_shared<OfrecordImageDecoderOpInterpCtxImpl<schema::OfrecordImageDecoderOp>>();
        ctx->set_name(name);
        ctx->set_color_space(color_space);
        return OpInterpUtil::Dispatch<Tensor>(*op, {input}, ctx);
      });
  m.add_functor("DispatchImageDecoderRandomCropResize",
                [](const std::shared_ptr<OpExpr>& op, const std::shared_ptr<Tensor> input,
                   int64_t target_width, int64_t target_height, int64_t seed, int64_t num_workers,
                   int64_t max_num_pixels, float random_area_min, float random_area_max,
                   float random_aspect_ratio_min, float random_aspect_ratio_max,
                   int64_t warmup_size, int64_t num_attempts) -> Maybe<Tensor> {
                  auto ctx = std::make_shared<ImageDecoderRandomCropResizeOpInterpCtx>();
                  ctx->target_width = target_width;
                  ctx->target_height = target_height;
                  ctx->seed = seed;
                  ctx->num_workers = num_workers;
                  ctx->max_num_pixels = max_num_pixels;
                  ctx->random_area_min = random_area_min;
                  ctx->random_area_max = random_area_max;
                  ctx->random_aspect_ratio_min = random_aspect_ratio_min;
                  ctx->random_aspect_ratio_max = random_aspect_ratio_max;
                  ctx->warmup_size = warmup_size;
                  ctx->num_attempts = num_attempts;
                  return OpInterpUtil::Dispatch<Tensor>(*op, {input}, ctx);
                });
  m.add_functor(
      "DispatchTensorBufferToListOfTensorsV2",
      [](const std::shared_ptr<OpExpr>& op, const std::shared_ptr<Tensor> input,
         const std::vector<Shape>& out_shapes, const std::vector<Symbol<DType>>& out_dtypes,
         bool dynamic_out) -> Maybe<TensorTuple> {
        auto ctx = std::make_shared<TensorBufferToListOfTensorsV2OpInterpCtxImpl<
            schema::TensorBufferToListOfTensorsV2Op>>();
        ctx->set_out_shapes(out_shapes);
        ctx->set_dynamic_out(dynamic_out);
        auto out_data_types = std::vector<DataType>();
        for (auto it = out_dtypes.begin(); it != out_dtypes.end(); it++) {
          out_data_types.emplace_back((*it)->data_type());
        }
        ctx->set_out_dtypes(out_data_types);
        return OpInterpUtil::Dispatch<TensorTuple>(*op, {input}, ctx);
      });
  m.add_functor(
      "DispatchImageResizeKeepAspectRatio",
      [](const std::shared_ptr<OpExpr>& op, const std::shared_ptr<Tensor> input,
         int32_t target_size, int32_t min_size, int32_t max_size, bool resize_longer,
         const std::string interpolation_type) -> Maybe<TensorTuple> {
        auto ctx = std::make_shared<
            ImageResizeKeepAspectRatioOpInterpCtxImpl<schema::ImageResizeKeepAspectRatioOp>>();
        ctx->set_target_size(target_size);
        ctx->set_min_size(min_size);
        ctx->set_max_size(max_size);
        ctx->set_resize_longer(resize_longer);
        ctx->set_interpolation_type(interpolation_type);
        return OpInterpUtil::Dispatch<TensorTuple>(*op, {input}, ctx);
      });
  m.add_functor(
      "DispatchImageResizeToFixed",
      [](const std::shared_ptr<OpExpr>& op, const std::shared_ptr<Tensor> input,
         int64_t target_width, int64_t target_height, int64_t channels,
         const Symbol<DType>& data_type,
         const std::string interpolation_type) -> Maybe<TensorTuple> {
        auto ctx =
            std::make_shared<ImageResizeToFixedOpInterpCtxImpl<schema::ImageResizeToFixedOp>>();
        ctx->set_target_width(target_width);
        ctx->set_target_height(target_height);
        ctx->set_channels(channels);
        ctx->set_data_type(data_type->data_type());
        ctx->set_interpolation_type(interpolation_type);
        return OpInterpUtil::Dispatch<TensorTuple>(*op, {input}, ctx);
      });
  m.add_functor("DispatchImageNormalize",
                [](const std::shared_ptr<OpExpr>& op, const std::shared_ptr<Tensor> input,
                   const std::vector<float>& mean, const std::vector<float>& std) -> Maybe<Tensor> {
                  auto ctx =
                      std::make_shared<ImageNormalizeOpInterpCtxImpl<schema::ImageNormalizeOp>>();
                  ctx->set_std(std);
                  ctx->set_mean(mean);
                  return OpInterpUtil::Dispatch<Tensor>(*op, {input}, ctx);
                });
  m.add_functor("DispatchCOCOReader",
                [](const std::shared_ptr<OpExpr>& op, const std::string image_dir,
                   const std::string annotation_file, int64_t batch_size, bool shuffle_after_epoch,
                   int64_t random_seed, bool group_by_ratio, bool remove_images_without_annotations,
                   bool stride_partition, int64_t session_id,
                   const Optional<Symbol<Device>>& device) -> Maybe<TensorTuple> {
                  auto ctx = std::make_shared<COCOReaderOpInterpCtxImpl<schema::COCOReaderOp>>();
                  ctx->set_session_id(session_id);
                  ctx->set_annotation_file(annotation_file);
                  ctx->set_image_dir(image_dir);
                  ctx->set_batch_size(batch_size);
                  ctx->set_shuffle_after_epoch(shuffle_after_epoch);
                  ctx->set_random_seed(random_seed);
                  ctx->set_group_by_ratio(group_by_ratio);
                  ctx->set_remove_images_without_annotations(remove_images_without_annotations);
                  ctx->set_stride_partition(stride_partition);
                  ctx->device = device;
                  return OpInterpUtil::Dispatch<TensorTuple>(*op, {}, ctx);
                });
  m.add_functor("DispatchCOCOReader",
                [](const std::shared_ptr<OpExpr>& op, const std::string image_dir,
                   const std::string annotation_file, int64_t batch_size, bool shuffle_after_epoch,
                   int64_t random_seed, bool group_by_ratio, bool remove_images_without_annotations,
                   bool stride_partition, int64_t session_id, const Symbol<ParallelDesc>& placement,
                   const std::vector<Symbol<cfg::SbpParallel>>& sbp_tuple) -> Maybe<TensorTuple> {
                  auto ctx = std::make_shared<COCOReaderOpInterpCtxImpl<schema::COCOReaderOp>>();
                  ctx->set_session_id(session_id);
                  ctx->set_annotation_file(annotation_file);
                  ctx->set_image_dir(image_dir);
                  ctx->set_batch_size(batch_size);
                  ctx->set_shuffle_after_epoch(shuffle_after_epoch);
                  ctx->set_random_seed(random_seed);
                  ctx->set_group_by_ratio(group_by_ratio);
                  ctx->set_remove_images_without_annotations(remove_images_without_annotations);
                  ctx->set_stride_partition(stride_partition);
                  ctx->set_nd_sbp(*JUST(GetNdSbpStrList(sbp_tuple)));
                  ctx->parallel_desc = placement;
                  ctx->sbp = JUST(GetNdSbp(sbp_tuple));
                  return OpInterpUtil::Dispatch<TensorTuple>(*op, {}, ctx);
                });
  m.add_functor(
      "DispatchImageBatchAlign",
      [](const std::shared_ptr<OpExpr>& op, const std::shared_ptr<Tensor> input, int32_t alignment,
         const Shape& shape, const Symbol<DType>& data_type, bool dynamic_out) -> Maybe<Tensor> {
        auto ctx = std::make_shared<ImageBatchAlignOpInterpCtxImpl<schema::ImageBatchAlignOp>>();
        ctx->set_shape(shape);
        ctx->set_data_type(data_type->data_type());
        ctx->set_alignment(alignment);
        ctx->set_dynamic_out(dynamic_out);
        return OpInterpUtil::Dispatch<Tensor>(*op, {input}, ctx);
      });
  m.add_functor(
      "DispatchOfrecordBytesDecoder",
      [](const std::shared_ptr<OpExpr>& op, const std::shared_ptr<Tensor> input,
         const std::string name) -> Maybe<Tensor> {
        auto ctx =
            std::make_shared<OfrecordBytesDecoderOpInterpCtxImpl<schema::OfrecordBytesDecoderOp>>();
        ctx->set_name(name);
        return OpInterpUtil::Dispatch<Tensor>(*op, {input}, ctx);
      });
  m.add_functor(
      "DispatchMegatronGptMmapDataLoader",
      [](const std::shared_ptr<OpExpr>& op, const std::string data_file_prefix, int64_t seq_length,
         int64_t label_length, int64_t num_samples, int64_t batch_size, const Symbol<DType>& dtype,
         const std::vector<int64_t>& split_sizes, int64_t split_index, bool shuffle,
         int64_t random_seed, const Optional<Symbol<Device>>& device) -> Maybe<Tensor> {
        auto ctx = std::make_shared<
            MegatronGptMmapDataLoaderOpInterpCtxImpl<schema::MegatronGptMmapDataLoaderOp>>();
        ctx->set_data_file_prefix(data_file_prefix);
        ctx->set_seq_length(seq_length);
        ctx->set_label_length(label_length);
        ctx->set_num_samples(num_samples);
        ctx->set_batch_size(batch_size);
        ctx->set_dtype(dtype->data_type());
        ctx->set_split_sizes(split_sizes);
        ctx->set_split_index(split_index);
        ctx->set_shuffle(shuffle);
        ctx->set_random_seed(random_seed);
        ctx->device = device;
        return OpInterpUtil::Dispatch<Tensor>(*op, {}, ctx);
      });
  m.add_functor(
      "DispatchMegatronGptMmapDataLoader",
      [](const std::shared_ptr<OpExpr>& op, const std::string data_file_prefix, int64_t seq_length,
         int64_t label_length, int64_t num_samples, int64_t batch_size, const Symbol<DType>& dtype,
         const std::vector<int64_t>& split_sizes, int64_t split_index, bool shuffle,
         int64_t random_seed, const Symbol<ParallelDesc>& placement,
         const std::vector<Symbol<cfg::SbpParallel>>& sbp_tuple) -> Maybe<Tensor> {
        auto ctx = std::make_shared<
            MegatronGptMmapDataLoaderOpInterpCtxImpl<schema::MegatronGptMmapDataLoaderOp>>();
        ctx->set_data_file_prefix(data_file_prefix);
        ctx->set_seq_length(seq_length);
        ctx->set_label_length(label_length);
        ctx->set_num_samples(num_samples);
        ctx->set_batch_size(batch_size);
        ctx->set_dtype(dtype->data_type());
        ctx->set_split_sizes(split_sizes);
        ctx->set_split_index(split_index);
        ctx->set_shuffle(shuffle);
        ctx->set_random_seed(random_seed);
        ctx->set_nd_sbp(*JUST(GetNdSbpStrList(sbp_tuple)));
        ctx->parallel_desc = placement;
        ctx->sbp = JUST(GetNdSbp(sbp_tuple));
        return OpInterpUtil::Dispatch<Tensor>(*op, {}, ctx);
      });
}

}  // namespace impl

}  // namespace functional
}  // namespace one
}  // namespace oneflow
