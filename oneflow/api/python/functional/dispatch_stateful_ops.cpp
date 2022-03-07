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
#include "oneflow/core/framework/attr_map.h"
#include "oneflow/core/framework/nd_sbp.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
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
        return OpInterpUtil::Dispatch<Tensor>(*op, {input});
      });
  m.add_functor(
      "DispatchFetchOutput",
      [](const std::shared_ptr<OpExpr>& op, const std::shared_ptr<Tensor>& input) -> Maybe<Tensor> {
        return OpInterpUtil::Dispatch<Tensor>(*op, {input});
      });
  m.add_functor("DispatchFeedVariable",
                [](const std::shared_ptr<OpExpr>& op, const std::shared_ptr<Tensor>& input,
                   const Scalar& l2) -> Maybe<Tensor> {
                  MutableAttrMap attrs;
                  JUST(attrs.SetAttr<double>("l2", JUST(l2.As<double>())));
                  return OpInterpUtil::Dispatch<Tensor>(*op, {input}, attrs);
                });
  m.add_functor(
      "DispatchOfrecordReader",
      [](const std::shared_ptr<OpExpr>& op, const std::string& data_dir, int32_t data_part_num,
         const std::string& part_name_prefix, int32_t part_name_suffix_length, int32_t batch_size,
         int32_t shuffle_buffer_size, bool random_shuffle, bool shuffle_after_epoch, int64_t seed,
         const Optional<Symbol<Device>>& device) -> Maybe<Tensor> {
        MutableAttrMap attrs;
        JUST(attrs.SetAttr("data_dir", data_dir));
        JUST(attrs.SetAttr("data_part_num", data_part_num));
        JUST(attrs.SetAttr("part_name_prefix", part_name_prefix));
        JUST(attrs.SetAttr("part_name_suffix_length", part_name_suffix_length));
        JUST(attrs.SetAttr("batch_size", batch_size));
        JUST(attrs.SetAttr("shuffle_buffer_size", shuffle_buffer_size));
        JUST(attrs.SetAttr("random_shuffle", random_shuffle));
        JUST(attrs.SetAttr("shuffle_after_epoch", shuffle_after_epoch));
        JUST(attrs.SetAttr("seed", seed));
        return OpInterpUtil::Dispatch<Tensor>(*op, {}, OpExprInterpContext(attrs, JUST(device)));
      });
  m.add_functor(
      "DispatchOfrecordReader",
      [](const std::shared_ptr<OpExpr>& op, const std::string& data_dir, int32_t data_part_num,
         const std::string& part_name_prefix, int32_t part_name_suffix_length, int32_t batch_size,
         int32_t shuffle_buffer_size, bool random_shuffle, bool shuffle_after_epoch, int64_t seed,
         const Symbol<ParallelDesc>& placement,
         const std::vector<Symbol<SbpParallel>>& sbp_tuple) -> Maybe<Tensor> {
        MutableAttrMap attrs;
        JUST(attrs.SetAttr("data_dir", data_dir));
        JUST(attrs.SetAttr("data_part_num", data_part_num));
        JUST(attrs.SetAttr("part_name_prefix", part_name_prefix));
        JUST(attrs.SetAttr("part_name_suffix_length", part_name_suffix_length));
        JUST(attrs.SetAttr("batch_size", batch_size));
        JUST(attrs.SetAttr("shuffle_buffer_size", shuffle_buffer_size));
        JUST(attrs.SetAttr("random_shuffle", random_shuffle));
        JUST(attrs.SetAttr("shuffle_after_epoch", shuffle_after_epoch));
        JUST(attrs.SetAttr("seed", seed));
        JUST(attrs.SetAttr("nd_sbp", *JUST(GetNdSbpStrList(sbp_tuple))));
        auto nd_sbp = JUST(GetNdSbp(sbp_tuple));
        return OpInterpUtil::Dispatch<Tensor>(*op, {},
                                              OpExprInterpContext(attrs, placement, nd_sbp));
      });
  m.add_functor("DispatchOfrecordRawDecoder",
                [](const std::shared_ptr<OpExpr>& op, const std::shared_ptr<Tensor>& input,
                   const std::string& name, const Shape& shape, const Symbol<DType>& data_type,
                   bool dim1_varying_length, bool truncate) -> Maybe<Tensor> {
                  MutableAttrMap attrs;
                  JUST(attrs.SetAttr("name", name));
                  JUST(attrs.SetAttr("shape", shape));
                  JUST(attrs.SetAttr("data_type", data_type->data_type()));
                  JUST(attrs.SetAttr("dim1_varying_length", dim1_varying_length));
                  JUST(attrs.SetAttr("truncate", truncate));
                  return OpInterpUtil::Dispatch<Tensor>(*op, {input}, attrs);
                });
  m.add_functor(
      "DispatchCoinFlip",
      [](const std::shared_ptr<OpExpr>& op, int64_t batch_size, Scalar probability, int64_t seed,
         bool has_seed, const Optional<Symbol<Device>>& device) -> Maybe<Tensor> {
        MutableAttrMap attrs;
        JUST(attrs.SetAttr("probability", JUST(probability.As<float>())));
        JUST(attrs.SetAttr("batch_size", batch_size));
        JUST(attrs.SetAttr("seed", seed));
        JUST(attrs.SetAttr("has_seed", has_seed));
        return OpInterpUtil::Dispatch<Tensor>(*op, {}, OpExprInterpContext(attrs, JUST(device)));
      });
  m.add_functor("DispatchCoinFlip",
                [](const std::shared_ptr<OpExpr>& op, int64_t batch_size, Scalar probability,
                   int64_t seed, bool has_seed, const Symbol<ParallelDesc>& placement,
                   const std::vector<Symbol<SbpParallel>>& sbp_tuple) -> Maybe<Tensor> {
                  MutableAttrMap attrs;
                  JUST(attrs.SetAttr("probability", JUST(probability.As<float>())));
                  JUST(attrs.SetAttr("batch_size", batch_size));
                  JUST(attrs.SetAttr("seed", seed));
                  JUST(attrs.SetAttr("has_seed", has_seed));
                  JUST(attrs.SetAttr("nd_sbp", *JUST(GetNdSbpStrList(sbp_tuple))));
                  auto nd_sbp = JUST(GetNdSbp(sbp_tuple));
                  return OpInterpUtil::Dispatch<Tensor>(
                      *op, {}, OpExprInterpContext(attrs, placement, nd_sbp));
                });
  m.add_functor(
      "DispatchDistributedPariticalFCSample",
      [](const std::shared_ptr<OpExpr>& op, const std::shared_ptr<Tensor>& weight,
         const std::shared_ptr<Tensor>& label, const int64_t& num_sample) -> Maybe<TensorTuple> {
        MutableAttrMap attrs;
        JUST(attrs.SetAttr<int64_t>("num_sample", num_sample));
        return OpInterpUtil::Dispatch<TensorTuple>(*op, {weight, label}, attrs);
      });
  m.add_functor(
      "DispatchCropMirrorNormalizeFromUint8",
      [](const std::shared_ptr<OpExpr>& op, const TensorTuple& input, int64_t crop_h,
         int64_t crop_w, float crop_pos_x, float crop_pos_y, const std::vector<float>& mean,
         const std::vector<float>& std, const Symbol<DType>& output_dtype,
         const std::string& output_layout, const std::string& color_space) -> Maybe<Tensor> {
        MutableAttrMap attrs;
        JUST(attrs.SetAttr("color_space", color_space));
        JUST(attrs.SetAttr("output_layout", output_layout));
        JUST(attrs.SetAttr("mean", mean));
        JUST(attrs.SetAttr("std", std));
        JUST(attrs.SetAttr("crop_h", crop_h));
        JUST(attrs.SetAttr("crop_w", crop_w));
        JUST(attrs.SetAttr("crop_pos_x", crop_pos_x));
        JUST(attrs.SetAttr("crop_pos_y", crop_pos_y));
        JUST(attrs.SetAttr("output_dtype", output_dtype->data_type()));
        return OpInterpUtil::Dispatch<Tensor>(*op, input, attrs);
      });
  m.add_functor(
      "DispatchCropMirrorNormalizeFromTensorBuffer",
      [](const std::shared_ptr<OpExpr>& op, const TensorTuple& input, int64_t crop_h,
         int64_t crop_w, float crop_pos_x, float crop_pos_y, const std::vector<float>& mean,
         const std::vector<float>& std, const Symbol<DType>& output_dtype,
         const std::string& output_layout, const std::string& color_space) -> Maybe<Tensor> {
        MutableAttrMap attrs;
        JUST(attrs.SetAttr("color_space", color_space));
        JUST(attrs.SetAttr("output_layout", output_layout));
        JUST(attrs.SetAttr("mean", mean));
        JUST(attrs.SetAttr("std", std));
        JUST(attrs.SetAttr("crop_h", crop_h));
        JUST(attrs.SetAttr("crop_w", crop_w));
        JUST(attrs.SetAttr("crop_pos_x", crop_pos_x));
        JUST(attrs.SetAttr("crop_pos_y", crop_pos_y));
        JUST(attrs.SetAttr("output_dtype", output_dtype->data_type()));
        return OpInterpUtil::Dispatch<Tensor>(*op, {input}, attrs);
      });
  m.add_functor(
      "DispatchOfrecordImageDecoderRandomCrop",
      [](const std::shared_ptr<OpExpr>& op, const std::shared_ptr<Tensor>& input,
         const std::string& name, const std::string& color_space,
         const std::vector<float>& random_area, const std::vector<float>& random_aspect_ratio,
         int32_t num_attempts, int64_t seed, bool has_seed) -> Maybe<Tensor> {
        MutableAttrMap attrs;
        JUST(attrs.SetAttr("name", name));
        JUST(attrs.SetAttr("color_space", color_space));
        JUST(attrs.SetAttr("num_attempts", num_attempts));
        JUST(attrs.SetAttr("seed", seed));
        JUST(attrs.SetAttr("has_seed", has_seed));
        JUST(attrs.SetAttr("random_area", random_area));
        JUST(attrs.SetAttr("random_aspect_ratio", random_aspect_ratio));
        return OpInterpUtil::Dispatch<Tensor>(*op, {input}, attrs);
      });
  m.add_functor("DispatchOfrecordImageDecoder",
                [](const std::shared_ptr<OpExpr>& op, const std::shared_ptr<Tensor>& input,
                   const std::string& name, const std::string& color_space) -> Maybe<Tensor> {
                  MutableAttrMap attrs;
                  JUST(attrs.SetAttr("name", name));
                  JUST(attrs.SetAttr("color_space", color_space));
                  return OpInterpUtil::Dispatch<Tensor>(*op, {input}, attrs);
                });
  m.add_functor("DispatchImageDecoderRandomCropResize",
                [](const std::shared_ptr<OpExpr>& op, const std::shared_ptr<Tensor>& input,
                   int64_t target_width, int64_t target_height, int64_t seed, int64_t num_workers,
                   int64_t max_num_pixels, float random_area_min, float random_area_max,
                   float random_aspect_ratio_min, float random_aspect_ratio_max,
                   int64_t warmup_size, int64_t num_attempts) -> Maybe<Tensor> {
                  MutableAttrMap attrs;
                  JUST(attrs.SetAttr("target_width", target_width));
                  JUST(attrs.SetAttr("target_height", target_height));
                  JUST(attrs.SetAttr("seed", seed));
                  JUST(attrs.SetAttr("num_workers", num_workers));
                  JUST(attrs.SetAttr("max_num_pixels", max_num_pixels));
                  JUST(attrs.SetAttr("random_area_min", random_area_min));
                  JUST(attrs.SetAttr("random_area_max", random_area_max));
                  JUST(attrs.SetAttr("random_aspect_ratio_min", random_aspect_ratio_min));
                  JUST(attrs.SetAttr("random_aspect_ratio_max", random_aspect_ratio_max));
                  JUST(attrs.SetAttr("warmup_size", warmup_size));
                  JUST(attrs.SetAttr("num_attempts", num_attempts));
                  return OpInterpUtil::Dispatch<Tensor>(*op, {input}, attrs);
                });
  m.add_functor(
      "DispatchTensorBufferToListOfTensorsV2",
      [](const std::shared_ptr<OpExpr>& op, const std::shared_ptr<Tensor>& input,
         const std::vector<Shape>& out_shapes, const std::vector<Symbol<DType>>& out_dtypes,
         bool dynamic_out) -> Maybe<TensorTuple> {
        MutableAttrMap attrs;
        JUST(attrs.SetAttr("out_shapes", out_shapes));
        JUST(attrs.SetAttr("dynamic_out", dynamic_out));
        auto out_data_types = std::vector<DataType>();
        for (auto it = out_dtypes.begin(); it != out_dtypes.end(); it++) {
          out_data_types.emplace_back((*it)->data_type());
        }
        JUST(attrs.SetAttr("out_dtypes", out_data_types));
        return OpInterpUtil::Dispatch<TensorTuple>(*op, {input}, attrs);
      });
  m.add_functor("DispatchImageResizeKeepAspectRatio",
                [](const std::shared_ptr<OpExpr>& op, const std::shared_ptr<Tensor>& input,
                   int32_t target_size, int32_t min_size, int32_t max_size, bool resize_longer,
                   const std::string& interpolation_type) -> Maybe<TensorTuple> {
                  MutableAttrMap attrs;
                  JUST(attrs.SetAttr("target_size", target_size));
                  JUST(attrs.SetAttr("min_size", min_size));
                  JUST(attrs.SetAttr("max_size", max_size));
                  JUST(attrs.SetAttr("resize_longer", resize_longer));
                  JUST(attrs.SetAttr("interpolation_type", interpolation_type));
                  return OpInterpUtil::Dispatch<TensorTuple>(*op, {input}, attrs);
                });
  m.add_functor("DispatchImageResizeToFixed",
                [](const std::shared_ptr<OpExpr>& op, const std::shared_ptr<Tensor>& input,
                   int64_t target_width, int64_t target_height, int64_t channels,
                   const Symbol<DType>& data_type,
                   const std::string& interpolation_type) -> Maybe<TensorTuple> {
                  MutableAttrMap attrs;
                  JUST(attrs.SetAttr("target_width", target_width));
                  JUST(attrs.SetAttr("target_height", target_height));
                  JUST(attrs.SetAttr("channels", channels));
                  JUST(attrs.SetAttr("data_type", data_type->data_type()));
                  JUST(attrs.SetAttr("interpolation_type", interpolation_type));
                  return OpInterpUtil::Dispatch<TensorTuple>(*op, {input}, attrs);
                });
  m.add_functor(
      "DispatchImageDecode",
      [](const std::shared_ptr<OpExpr>& op, const std::shared_ptr<Tensor>& input,
         const std::string& color_space, const Symbol<DType>& data_type) -> Maybe<Tensor> {
        MutableAttrMap attrs;
        JUST(attrs.SetAttr("color_space", color_space));
        JUST(attrs.SetAttr("data_type", data_type->data_type()));
        return OpInterpUtil::Dispatch<Tensor>(*op, {input}, attrs);
      });
  m.add_functor("DispatchImageNormalize",
                [](const std::shared_ptr<OpExpr>& op, const std::shared_ptr<Tensor>& input,
                   const std::vector<float>& mean, const std::vector<float>& std) -> Maybe<Tensor> {
                  MutableAttrMap attrs;
                  JUST(attrs.SetAttr("std", std));
                  JUST(attrs.SetAttr("mean", mean));
                  return OpInterpUtil::Dispatch<Tensor>(*op, {input}, attrs);
                });
  m.add_functor(
      "DispatchCOCOReader",
      [](const std::shared_ptr<OpExpr>& op, const std::string& image_dir,
         const std::string& annotation_file, int64_t batch_size, bool shuffle_after_epoch,
         int64_t random_seed, bool group_by_ratio, bool remove_images_without_annotations,
         bool stride_partition, int64_t session_id,
         const Optional<Symbol<Device>>& device) -> Maybe<TensorTuple> {
        MutableAttrMap attrs;
        JUST(attrs.SetAttr("session_id", session_id));
        JUST(attrs.SetAttr("annotation_file", annotation_file));
        JUST(attrs.SetAttr("image_dir", image_dir));
        JUST(attrs.SetAttr("batch_size", batch_size));
        JUST(attrs.SetAttr("shuffle_after_epoch", shuffle_after_epoch));
        JUST(attrs.SetAttr("random_seed", random_seed));
        JUST(attrs.SetAttr("group_by_ratio", group_by_ratio));
        JUST(attrs.SetAttr("remove_images_without_annotations", remove_images_without_annotations));
        JUST(attrs.SetAttr("stride_partition", stride_partition));
        return OpInterpUtil::Dispatch<TensorTuple>(*op, {},
                                                   OpExprInterpContext(attrs, JUST(device)));
      });
  m.add_functor(
      "DispatchCOCOReader",
      [](const std::shared_ptr<OpExpr>& op, const std::string& image_dir,
         const std::string& annotation_file, int64_t batch_size, bool shuffle_after_epoch,
         int64_t random_seed, bool group_by_ratio, bool remove_images_without_annotations,
         bool stride_partition, int64_t session_id, const Symbol<ParallelDesc>& placement,
         const std::vector<Symbol<SbpParallel>>& sbp_tuple) -> Maybe<TensorTuple> {
        MutableAttrMap attrs;
        JUST(attrs.SetAttr("session_id", session_id));
        JUST(attrs.SetAttr("annotation_file", annotation_file));
        JUST(attrs.SetAttr("image_dir", image_dir));
        JUST(attrs.SetAttr("batch_size", batch_size));
        JUST(attrs.SetAttr("shuffle_after_epoch", shuffle_after_epoch));
        JUST(attrs.SetAttr("random_seed", random_seed));
        JUST(attrs.SetAttr("group_by_ratio", group_by_ratio));
        JUST(attrs.SetAttr("remove_images_without_annotations", remove_images_without_annotations));
        JUST(attrs.SetAttr("stride_partition", stride_partition));
        JUST(attrs.SetAttr("nd_sbp", *JUST(GetNdSbpStrList(sbp_tuple))));
        auto nd_sbp = JUST(GetNdSbp(sbp_tuple));
        return OpInterpUtil::Dispatch<TensorTuple>(*op, {},
                                                   OpExprInterpContext(attrs, placement, nd_sbp));
      });
  m.add_functor(
      "DispatchImageBatchAlign",
      [](const std::shared_ptr<OpExpr>& op, const std::shared_ptr<Tensor>& input, int32_t alignment,
         const Shape& shape, const Symbol<DType>& data_type, bool dynamic_out) -> Maybe<Tensor> {
        MutableAttrMap attrs;
        JUST(attrs.SetAttr("shape", shape));
        JUST(attrs.SetAttr("data_type", data_type->data_type()));
        JUST(attrs.SetAttr("alignment", alignment));
        JUST(attrs.SetAttr("dynamic_out", dynamic_out));
        return OpInterpUtil::Dispatch<Tensor>(*op, {input}, attrs);
      });
  m.add_functor("DispatchOfrecordBytesDecoder",
                [](const std::shared_ptr<OpExpr>& op, const std::shared_ptr<Tensor>& input,
                   const std::string& name) -> Maybe<Tensor> {
                  MutableAttrMap attrs;
                  JUST(attrs.SetAttr("name", name));
                  return OpInterpUtil::Dispatch<Tensor>(*op, {input}, attrs);
                });
  m.add_functor(
      "DispatchOneRecReader",
      [](const std::shared_ptr<OpExpr>& op, const std::vector<std::string>& files,
         const int64_t batch_size, const bool random_shuffle, const std::string& shuffle_mode,
         const int32_t shuffle_buffer_size, const bool shuffle_after_epoch, int64_t random_seed,
         const bool verify_example, const Optional<Symbol<Device>>& device) -> Maybe<Tensor> {
        MutableAttrMap attrs;
        JUST(attrs.SetAttr<std::vector<std::string>>("files", files));
        JUST(attrs.SetAttr<int64_t>("batch_size", batch_size));
        JUST(attrs.SetAttr<bool>("random_shuffle", random_shuffle));
        JUST(attrs.SetAttr<std::string>("shuffle_mode", shuffle_mode));
        JUST(attrs.SetAttr<int32_t>("shuffle_buffer_size", shuffle_buffer_size));
        JUST(attrs.SetAttr<bool>("shuffle_after_epoch", shuffle_after_epoch));
        JUST(attrs.SetAttr<int64_t>("seed", random_seed));
        JUST(attrs.SetAttr<bool>("verify_example", verify_example));
        return OpInterpUtil::Dispatch<Tensor>(*op, {}, OpExprInterpContext(attrs, JUST(device)));
      });
  m.add_functor(
      "DispatchOneRecReader",
      [](const std::shared_ptr<OpExpr>& op, const std::vector<std::string>& files,
         const int64_t batch_size, const bool random_shuffle, const std::string& shuffle_mode,
         const int32_t shuffle_buffer_size, const bool shuffle_after_epoch, int64_t random_seed,
         const bool verify_example, const Symbol<ParallelDesc>& placement,
         const std::vector<Symbol<SbpParallel>>& sbp_tuple) -> Maybe<Tensor> {
        MutableAttrMap attrs;
        JUST(attrs.SetAttr<std::vector<std::string>>("files", files));
        JUST(attrs.SetAttr<int64_t>("batch_size", batch_size));
        JUST(attrs.SetAttr<bool>("random_shuffle", random_shuffle));
        JUST(attrs.SetAttr<std::string>("shuffle_mode", shuffle_mode));
        JUST(attrs.SetAttr<int32_t>("shuffle_buffer_size", shuffle_buffer_size));
        JUST(attrs.SetAttr<bool>("shuffle_after_epoch", shuffle_after_epoch));
        JUST(attrs.SetAttr<int64_t>("seed", random_seed));
        JUST(attrs.SetAttr<bool>("verify_example", verify_example));
        JUST(attrs.SetAttr("nd_sbp", *JUST(GetNdSbpStrList(sbp_tuple))));
        auto nd_sbp = JUST(GetNdSbp(sbp_tuple));
        return OpInterpUtil::Dispatch<Tensor>(*op, {},
                                              OpExprInterpContext(attrs, placement, nd_sbp));
      });
  m.add_functor(
      "DispatchMegatronGptMmapDataLoader",
      [](const std::shared_ptr<OpExpr>& op, const std::string& data_file_prefix, int64_t seq_length,
         int64_t label_length, int64_t num_samples, int64_t batch_size, const Symbol<DType>& dtype,
         const std::vector<int64_t>& split_sizes, int64_t split_index, bool shuffle,
         int64_t random_seed, const Optional<Symbol<Device>>& device) -> Maybe<Tensor> {
        MutableAttrMap attrs;
        JUST(attrs.SetAttr("data_file_prefix", data_file_prefix));
        JUST(attrs.SetAttr("seq_length", seq_length));
        JUST(attrs.SetAttr("label_length", label_length));
        JUST(attrs.SetAttr("num_samples", num_samples));
        JUST(attrs.SetAttr("batch_size", batch_size));
        JUST(attrs.SetAttr("dtype", dtype->data_type()));
        JUST(attrs.SetAttr("split_sizes", split_sizes));
        JUST(attrs.SetAttr("split_index", split_index));
        JUST(attrs.SetAttr("shuffle", shuffle));
        JUST(attrs.SetAttr("random_seed", random_seed));
        return OpInterpUtil::Dispatch<Tensor>(*op, {}, OpExprInterpContext(attrs, JUST(device)));
      });
  m.add_functor(
      "DispatchMegatronGptMmapDataLoader",
      [](const std::shared_ptr<OpExpr>& op, const std::string& data_file_prefix, int64_t seq_length,
         int64_t label_length, int64_t num_samples, int64_t batch_size, const Symbol<DType>& dtype,
         const std::vector<int64_t>& split_sizes, int64_t split_index, bool shuffle,
         int64_t random_seed, const Symbol<ParallelDesc>& placement,
         const std::vector<Symbol<SbpParallel>>& sbp_tuple) -> Maybe<Tensor> {
        MutableAttrMap attrs;
        JUST(attrs.SetAttr("data_file_prefix", data_file_prefix));
        JUST(attrs.SetAttr("seq_length", seq_length));
        JUST(attrs.SetAttr("label_length", label_length));
        JUST(attrs.SetAttr("num_samples", num_samples));
        JUST(attrs.SetAttr("batch_size", batch_size));
        JUST(attrs.SetAttr("dtype", dtype->data_type()));
        JUST(attrs.SetAttr("split_sizes", split_sizes));
        JUST(attrs.SetAttr("split_index", split_index));
        JUST(attrs.SetAttr("shuffle", shuffle));
        JUST(attrs.SetAttr("random_seed", random_seed));
        auto nd_sbp = JUST(GetNdSbp(sbp_tuple));
        return OpInterpUtil::Dispatch<Tensor>(*op, {},
                                              OpExprInterpContext(attrs, placement, nd_sbp));
      });
  m.add_functor("DispatchRmspropUpdate",
                [](const std::shared_ptr<OpExpr>& op, const TensorTuple& inputs,
                   float learning_rate, double scale, float l1, float l2, bool centered,
                   float epsilon, float decay_rate, float weight_decay) -> Maybe<void> {
                  MutableAttrMap attrs;
                  JUST(attrs.SetAttr("learning_rate_val", learning_rate));
                  JUST(attrs.SetAttr("scale", scale));
                  JUST(attrs.SetAttr("l1", l1));
                  JUST(attrs.SetAttr("l2", l2));
                  JUST(attrs.SetAttr("centered", centered));
                  JUST(attrs.SetAttr("epsilon", epsilon));
                  JUST(attrs.SetAttr("decay_rate", decay_rate));
                  JUST(attrs.SetAttr("weight_decay", weight_decay));
                  JUST(OpInterpUtil::Dispatch<TensorTuple>(*op, inputs, attrs));
                  return Maybe<void>::Ok();
                });
  m.add_functor("DispatchAdamUpdate",
                [](const std::shared_ptr<OpExpr>& op, const TensorTuple& inputs,
                   float learning_rate, float bias_correction1, float bias_correction2,
                   double scale, float l1, float l2, float beta1, float beta2, float epsilon,
                   float weight_decay, bool amsgrad, bool do_bias_correction) -> Maybe<void> {
                  MutableAttrMap attrs;
                  JUST(attrs.SetAttr("learning_rate_val", learning_rate));
                  JUST(attrs.SetAttr("bias_correction1_val", bias_correction1));
                  JUST(attrs.SetAttr("bias_correction2_val", bias_correction2));
                  JUST(attrs.SetAttr("scale", scale));
                  JUST(attrs.SetAttr("l1", l1));
                  JUST(attrs.SetAttr("l2", l2));
                  JUST(attrs.SetAttr("beta1", beta1));
                  JUST(attrs.SetAttr("beta2", beta2));
                  JUST(attrs.SetAttr("epsilon", epsilon));
                  JUST(attrs.SetAttr("weight_decay", weight_decay));
                  JUST(attrs.SetAttr("amsgrad", amsgrad));
                  JUST(attrs.SetAttr("do_bias_correction", do_bias_correction));
                  JUST(OpInterpUtil::Dispatch<TensorTuple>(*op, inputs, attrs));
                  return Maybe<void>::Ok();
                });
  m.add_functor("DispatchAdagradUpdate",
                [](const std::shared_ptr<OpExpr>& op, const TensorTuple& inputs,
                   float learning_rate, double scale, float l1, float l2, float lr_decay,
                   float weight_decay, float epsilon, int32_t train_step) -> Maybe<void> {
                  MutableAttrMap attrs;
                  JUST(attrs.SetAttr("learning_rate_val", learning_rate));
                  JUST(attrs.SetAttr("scale", scale));
                  JUST(attrs.SetAttr("l1", l1));
                  JUST(attrs.SetAttr("l2", l2));
                  JUST(attrs.SetAttr("lr_decay", lr_decay));
                  JUST(attrs.SetAttr("weight_decay", weight_decay));
                  JUST(attrs.SetAttr("epsilon", epsilon));
                  JUST(attrs.SetAttr("train_step_val", train_step));
                  JUST(OpInterpUtil::Dispatch<TensorTuple>(*op, inputs, attrs));
                  return Maybe<void>::Ok();
                });
  m.add_functor(
      "DispatchMomentumUpdate",
      [](const std::shared_ptr<OpExpr>& op, const TensorTuple& inputs, float learning_rate,
         double scale, float l1, float l2, float beta, float weight_decay) -> Maybe<void> {
        MutableAttrMap attrs;
        JUST(attrs.SetAttr("learning_rate_val", learning_rate));
        JUST(attrs.SetAttr("scale", scale));
        JUST(attrs.SetAttr("l1", l1));
        JUST(attrs.SetAttr("l2", l2));
        JUST(attrs.SetAttr("beta", beta));
        JUST(attrs.SetAttr("weight_decay", weight_decay));
        JUST(OpInterpUtil::Dispatch<TensorTuple>(*op, inputs, attrs));
        return Maybe<void>::Ok();
      });
  m.add_functor(
      "DispatchSgdUpdate",
      [](const std::shared_ptr<OpExpr>& op, const TensorTuple& inputs, float learning_rate,
         double scale, float l1, float l2, float weight_decay) -> Maybe<void> {
        MutableAttrMap attrs;
        JUST(attrs.SetAttr("learning_rate_val", learning_rate));
        JUST(attrs.SetAttr("scale", scale));
        JUST(attrs.SetAttr("l1", l1));
        JUST(attrs.SetAttr("l2", l2));
        JUST(attrs.SetAttr("weight_decay", weight_decay));
        JUST(OpInterpUtil::Dispatch<TensorTuple>(*op, inputs, attrs));
        return Maybe<void>::Ok();
      });
  m.add_functor("DispatchLambUpdate",
                [](const std::shared_ptr<OpExpr>& op, const TensorTuple& inputs,
                   float learning_rate, float bias_correction1, float bias_correction2,
                   double scale, float l1, float l2, float beta1, float beta2, float epsilon,
                   float weight_decay, bool do_bias_correction) -> Maybe<void> {
                  MutableAttrMap attrs;
                  JUST(attrs.SetAttr("learning_rate_val", learning_rate));
                  JUST(attrs.SetAttr("bias_correction1_val", bias_correction1));
                  JUST(attrs.SetAttr("bias_correction2_val", bias_correction2));
                  JUST(attrs.SetAttr("scale", scale));
                  JUST(attrs.SetAttr("l1", l1));
                  JUST(attrs.SetAttr("l2", l2));
                  JUST(attrs.SetAttr("beta1", beta1));
                  JUST(attrs.SetAttr("beta2", beta2));
                  JUST(attrs.SetAttr("epsilon", epsilon));
                  JUST(attrs.SetAttr("weight_decay", weight_decay));
                  JUST(attrs.SetAttr("do_bias_correction", do_bias_correction));
                  JUST(OpInterpUtil::Dispatch<TensorTuple>(*op, inputs, attrs));
                  return Maybe<void>::Ok();
                });
  m.add_functor("DispatchEagerNcclAllReduce",
                [](const std::shared_ptr<OpExpr>& op, const std::shared_ptr<Tensor>& input,
                   const std::string& parallel_conf, bool async_launch) -> Maybe<Tensor> {
                  MutableAttrMap attrs;
                  JUST(attrs.SetAttr("parallel_conf", parallel_conf));
                  JUST(attrs.SetAttr("async_launch", async_launch));
                  return OpInterpUtil::Dispatch<Tensor>(*op, {input}, attrs);
                });
}

}  // namespace impl

}  // namespace functional
}  // namespace one
}  // namespace oneflow
