#include "oneflow/core/kernel/bbox_nms_kernel.h"

namespace oneflow {

template<typename T>
void BboxNmsKernel<T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* bbox_blob = BnInOp2Blob("bbox");
  const Blob* score_blob = BnInOp2Blob("bbox_score");
  Blob* out_bbox_blob = BnInOp2Blob("out_bbox");
  Blob* out_score_blob = BnInOp2Blob("out_bbox_score");
  Blob* out_label_blob = BnInOp2Blob("out_bbox_label");
  const int32_t num_axes = score_blob->shape().NumAxes();

  std::vector<std::vector<int32_t>> grouped_im2class_id_vec;
  std::vector<std::vector<int32_t>> grouped_class2score_id_vec;
  if (num_axes == 2) {
    GroupByImRecordAndClass(NDIMS<2>(), score_blob, grouped_im2class_id_vec,
                            grouped_class2score_id_vec);
  } else if (num_axes == 3) {
    GroupByImRecordAndClass(NDIMS<3>(), score_blob, grouped_im2class_id_vec,
                            grouped_class2score_id_vec);
  } else {
    UNIMPLEMENTED();
  }

  auto grouped_class2bbox_id_vec = ApplyNms(score_blob, bbox_blob, grouped_class2score_id_vec);
  auto grouped_im2score_id_vec =
      ConcatAllClasses(grouped_im2class_id_vec, grouped_class2bbox_id_vec);

  if (num_axes == 2) {
    WriteToOutput(NDIMS<2>(), grouped_im2score_id_vec, bbox_blob, score_blob, out_bbox_blob,
                  out_score_blob, out_label_blob);
  } else if (num_axes == 3) {
    WriteToOutput(NDIMS<3>(), grouped_im2score_id_vec, bbox_blob, score_blob, out_bbox_blob,
                  out_score_blob, out_label_blob);
  } else {
    UNIMPLEMENTED();
  }
}

template<typename T>
void BboxNmsKernel<T>::GroupByImRecordAndClass(
    NDIMS<2>, const Blob* score_blob, std::vector<std::vector<int32_t>>& im2class_id_vec,
    std::vector<std::vector<int32_t>>& class2score_id_vec) const {
  const BboxNmsKernelConf& kernel_conf = this->kernel_conf().bbox_nms_conf();
  CHECK(kernel_conf.has_device_piece_size());
  int32_t num_records = kernel_conf.device_piece_size();
  im2class_id_vec.reserve(num_records);
  int32_t num_boxes = score_blob->shape().At(0);
  int32_t num_classes = score_blob->shape().At(1);
  class2score_id_vec.reserve(num_records * num_classes);
  FOR_RANGE(int32_t, i, 0, num_boxes) {
    int32_t record_id = score_blob->has_record_id_in_device_piece_field()
                            ? score_blob->record_id_in_device_piece(i)
                            : 0;
    im2class_id_vec[record_id].resize(num_classes);
    FOR_RANGE(int32_t, j, 0, num_classes) {
      int32_t class_id = record_id * num_classes + j;
      im2class_id_vec[record_id][j] = class_id;
      class2score_id_vec[class_id].emplace_back(i * num_classes + j);
    }
  }
}

template<typename T>
void BboxNmsKernel<T>::GroupByImRecordAndClass(
    NDIMS<3>, const Blob* score_blob, std::vector<std::vector<int32_t>>& im2class_id_vec,
    std::vector<std::vector<int32_t>>& class2score_id_vec) const {
  int32_t num_records = score_blob->shape().At(0);
  int32_t num_boxes = score_blob->shape().At(1);
  int32_t num_classes = score_blob->shape().At(2);
  im2class_id_vec.reserve(num_records);
  class2score_id_vec.reserve(num_records * num_classes);
  FOR_RANGE(int32_t, i, 0, num_records) {
    im2class_id_vec[i].resize(num_classes);
    FOR_RANGE(int32_t, k, 0, num_classes) {
      int32_t class_id = i * num_classes + k;
      im2class_id_vec[i][k] = class_id;
      int32_t valid_num_boxes =
          score_blob->has_dim1_valid_num_field() ? score_blob->dim1_valid_num(i) : num_boxes;
      class2score_id_vec[class_id].reserve(valid_num_boxes);
      FOR_RANGE(int32_t, j, 0, valid_num_boxes) {
        class2score_id_vec[class_id].emplace_back(i * num_boxes * num_classes + j * num_classes
                                                  + k);
      }
    }
  }
}

template<typename T>
std::vector<std::vector<int32_t>> BboxNmsKernel<T>::ApplyNms(
    const Blob* score_blob, const Blob* bbox_blob,
    const std::vector<std::vector<int32_t>>& class2score_id_vec) const {
  std::vector<std::vector<int32_t>> post_nms_class2bbox_id_vec(class2score_id_vec.size());
  const BboxNmsOpConf& op_conf = this->op_conf().bbox_nms_conf();
  const BboxNmsKernelConf& kernel_conf = this->kernel_conf().bbox_nms_conf();
  const T* score_ptr = score_blob->dptr<T>();
  const T* bbox_ptr = bbox_blob->dptr<T>();
  MultiThreadLoop(class2score_id_vec.size(), [&](int32_t i) {
    std::vector<int32_t> pre_nms_id_vec(class2score_id_vec[i]);
    if (op_conf.has_pre_nms_top_n()) {
      std::nth_element(pre_nms_id_vec.begin(), pre_nms_id_vec.begin() + op_conf.pre_nms_top_n(),
                       pre_nms_id_vec.end());
      pre_nms_id_vec.resize(op_conf.pre_nms_top_n());
    }
    std::sort(pre_nms_id_vec.begin(), pre_nms_id_vec.end(),
              [&](int32_t lid, int32_t rid) { return score_ptr[lid] > score_ptr[rid]; });
    if (op_conf.has_pre_nms_score_threshold()) {
      auto lt_thresh_it = std::find_if(
          pre_nms_id_vec.begin(), pre_nms_id_vec.end(),
          [&](int32_t id) { return score_ptr[id] < op_conf.pre_nms_score_threshold(); });
      pre_nms_id_vec.erase(lt_thresh_it, pre_nms_id_vec.end());
    }
    if (kernel_conf.need_broadcast()) {
      std::transform(pre_nms_id_vec.begin(), pre_nms_id_vec.end(), pre_nms_id_vec.begin(),
                     [&](int32_t id) { return id / kernel_conf.num_classes(); });
    }
    std::vector<int32_t>& post_nms_id_vec = post_nms_class2bbox_id_vec[i];
    size_t post_nms_top_n = pre_nms_id_vec.size();
    if (op_conf.has_post_nms_top_n()) {
      post_nms_top_n = std::min<size_t>(post_nms_top_n, op_conf.post_nms_top_n());
    }
    post_nms_id_vec.resize(post_nms_top_n);
    BBoxSlice pre_nms_slice(IndexSequence(pre_nms_id_vec.size(), pre_nms_id_vec.data(), false),
                            bbox_ptr);
    BBoxSlice post_nms_slice(IndexSequence(post_nms_id_vec.size(), post_nms_id_vec.data(), false),
                             bbox_ptr);
    BBoxUtil<BBox>::Nms(op_conf.nms_threshold(), pre_nms_slice, post_nms_slice);
    post_nms_id_vec.resize(post_nms_slice.size());
  });
  return post_nms_class2bbox_id_vec;
}

template<typename T>
std::vector<std::vector<int32_t>> BboxNmsKernel<T>::ConcatAllClasses(
    const std::vector<std::vector<int32_t>>& im2class_id_vec,
    const std::vector<std::vector<int32_t>>& class2bbox_id_vec) const {
  const BboxNmsOpConf& op_conf = this->op_conf().bbox_nms_conf();
  const BboxNmsKernelConf& kernel_conf = this->kernel_conf().bbox_nms_conf();
  std::vector<std::vector<int32_t>> im2score_id_vec(im2class_id_vec.size());
  FOR_RANGE(int32_t, i, 0, im2class_id_vec.size()) {
    std::vector<int32_t>& score_id_vec = im2score_id_vec[i];
    for (const int32_t class_id : im2class_id_vec[i]) {
      int32_t class_idx = class_id % kernel_conf.num_classes();
      for (const int32_t bbox_id : class2bbox_id_vec[class_id]) {
        int32_t score_id = kernel_conf.need_broadcast()
                               ? (bbox_id * kernel_conf.num_classes() + class_idx)
                               : bbox_id;
        score_id_vec.emplace_back(score_id);
      }
    }
    if (op_conf.has_image_top_n()) {
      if (op_conf.image_top_n() < score_id_vec.size()) {
        std::nth_element(score_id_vec.begin(), score_id_vec.begin() + op_conf.image_top_n(),
                         score_id_vec.end());
        score_id_vec.resize(op_conf.image_top_n());
      }
    }
  }
  return im2score_id_vec;
}

template<typename T>
void BboxNmsKernel<T>::WriteToOutput(NDIMS<2>,
                                     const std::vector<std::vector<int32_t>>& im2score_id_vec,
                                     const Blob* bbox_blob, const Blob* score_blob,
                                     Blob* out_bbox_blob, Blob* out_score_blob,
                                     Blob* out_label_blob) const {
  const BboxNmsKernelConf& kernel_conf = this->kernel_conf().bbox_nms_conf();
  int32_t num_output = 0;
  const BBox* bbox_ptr = BBox::Cast(bbox_blob->dptr<T>());
  BBox* out_bbox_ptr = BBox::Cast(out_bbox_blob->mut_dptr<T>());
  FOR_RANGE(int32_t, i, 0, im2score_id_vec.size()) {
    for (const int32_t score_id : im2score_id_vec[i]) {
      int32_t class_idx = score_id % kernel_conf.num_classes();
      int32_t bbox_id =
          kernel_conf.need_broadcast() ? score_id / kernel_conf.num_classes() : bbox_id;

      out_bbox_ptr[num_output].elem() = bbox_ptr[bbox_id].elem();
      out_score_blob->mut_dptr<T>()[num_output] = score_blob->dptr<T>()[score_id];
      if (out_label_blob) { out_label_blob->mut_dptr<int32_t>()[num_output] = class_idx; }
      if (bbox_blob->has_record_id_in_device_piece_field()) {
        out_bbox_blob->set_record_id_in_device_piece(num_output, i);
        out_score_blob->set_record_id_in_device_piece(num_output, i);
        if (out_label_blob) { out_label_blob->set_record_id_in_device_piece(num_output, i); }
      }
      ++num_output;
    }
  }
  out_bbox_blob->set_dim0_valid_num(0, num_output);
  out_score_blob->set_dim0_valid_num(0, num_output);
  if (out_label_blob) { out_label_blob->set_dim0_valid_num(0, num_output); }
}

template<typename T>
void BboxNmsKernel<T>::WriteToOutput(NDIMS<3>,
                                     const std::vector<std::vector<int32_t>>& im2score_id_vec,
                                     const Blob* bbox_blob, const Blob* score_blob,
                                     Blob* out_bbox_blob, Blob* out_score_blob,
                                     Blob* out_label_blob) const {
  const BboxNmsKernelConf& kernel_conf = this->kernel_conf().bbox_nms_conf();
  FOR_RANGE(int32_t, i, 0, im2score_id_vec.size()) {
    int32_t num_output_per_record = 0;
    const BBox* bbox_ptr = BBox::Cast(bbox_blob->dptr<T>());
    BBox* out_bbox_ptr = BBox::Cast(out_bbox_blob->mut_dptr<T>(i));
    for (const int32_t score_id : im2score_id_vec[i]) {
      int32_t class_idx = score_id % kernel_conf.num_classes();
      int32_t bbox_id =
          kernel_conf.need_broadcast() ? score_id / kernel_conf.num_classes() : bbox_id;

      out_bbox_ptr[num_output_per_record].elem() = bbox_ptr[bbox_id].elem();
      out_score_blob->mut_dptr<T>(i)[num_output_per_record] = score_blob->dptr<T>()[score_id];
      if (out_label_blob) {
        out_label_blob->mut_dptr<int32_t>(i)[num_output_per_record] = class_idx;
      }
      ++num_output_per_record;
    }
    out_bbox_blob->set_dim1_valid_num(i, num_output_per_record);
    out_score_blob->set_dim1_valid_num(i, num_output_per_record);
    if (out_label_blob) { out_label_blob->set_dim1_valid_num(i, num_output_per_record); }
  }
}

ADD_CPU_DEFAULT_KERNEL_CREATOR(OperatorConf::kBboxNmsConf, BboxNmsKernel, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow