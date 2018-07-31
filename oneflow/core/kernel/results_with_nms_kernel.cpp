#include "oneflow/core/kernel/results_with_nms_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"
#include <math.h>

namespace oneflow {

namespace {

template<typename T>
inline float BBoxArea(const T* box) {
  return (box[2] - box[0] + 1) * (box[3] - box[1] + 1);
}
template<typename T>
inline float InterOverUnion(const T* box0, const T* box1) {
  int32_t iw = std::min(box0[2], box1[2]) - std::max(box0[0], box1[0]) + 1;
  iw = std::max<int32_t>(iw, 0);
  int32_t ih = std::min(box0[3], box1[3]) - std::max(box0[1], box1[1]) + 1;
  ih = std::max<int32_t>(ih, 0);
  float inter = iw * ih;
  float ua = BBoxArea(box0) + BBoxArea(box1) - inter;
  return inter / ua;
}

template<typename T>
void SortScoreIndex(const int64_t img_proposal_num, const T* score_ptr,
                    int32_t* sorted_score_index_ptr) {
  FOR_RANGE(int64_t, i, 0, img_proposal_num) { sorted_score_index_ptr[i] = i; }
  std::sort(sorted_score_index_ptr, sorted_score_index_ptr + img_proposal_num,
            [&](int32_t lhs, int32_t rhs) { return score_ptr[lhs] > score_ptr[rhs]; });
}

template<typename T>
int32_t Nms(const T* const_img_proposal_ptr, const float nms_threshold,
            int32_t* sorted_score_index_ptr, const int32_t pre_nms_num,
            int32_t* supressed_index_ptr) {
  FOR_RANGE(int32_t, i, 0, pre_nms_num) { supressed_index_ptr[i] = false; }
  int32_t keep_num = 0;
  FOR_RANGE(int32_t, i, 0, pre_nms_num) {
    if (supressed_index_ptr[i]) { continue; }
    sorted_score_index_ptr[keep_num++] = sorted_score_index_ptr[i];
    const int32_t cur_proposal_offset = sorted_score_index_ptr[i] * 4;
    const T* cur_proposal_ptr = const_img_proposal_ptr + cur_proposal_offset;
    FOR_RANGE(int32_t, j, i + 1, pre_nms_num) {
      if (supressed_index_ptr[j]) { continue; }
      const int32_t cand_proposal_offset = sorted_score_index_ptr[j] * 4;
      const T* cand_proposal_ptr = const_img_proposal_ptr + cand_proposal_offset;
      if (InterOverUnion(cur_proposal_ptr, cand_proposal_ptr) >= nms_threshold) {
        supressed_index_ptr[j] = true;
      }
    }
  }
  return keep_num;
}

template<typename T>
void BboxVoting(int32_t class_index, int32_t nms_num, int32_t all_num,
                const BboxVoteConf& bbox_vote,
                std::function<Blob*(const std::string&)> BnInOp2Blob) {
  /*
  nms的输出个数不确定，只能设最大值box_num
  */
  const T* all_box = BnInOp2Blob("box_per_class")->dptr<T>();
  const T* all_score = BnInOp2Blob("score_per_class")->dptr<T>();
  int32_t box_num = BnInOp2Blob("boxes")->shape().At(1);
  int32_t nms_box_offset = class_index * box_num * 4;
  T* nms_box = BnInOp2Blob("nms_boxes")->mut_dptr<T>() + nms_box_offset;
  T* nms_score = BnInOp2Blob("nms_scores")->mut_dptr<T>() + nms_box_offset / 4;
  T* vot_boxes_inds = BnInOp2Blob("vot_boxes_inds")->mut_dptr<T>();
  const float beta = bbox_vote.beta();
  const int32_t thresh = bbox_vote.thresh();
  const std::string score_method = bbox_vote.score_method();
  FOR_RANGE(int32_t, i, 0, nms_num) {
    int32_t vt_idx = 0;
    T iou_weight_avg = 0;
    FOR_RANGE(int32_t, j, 0, all_num) {
      float iou = InterOverUnion(nms_box[i], all_box[j]);
      if (iou >= thresh) {
        vot_boxes_inds[vt_idx++] = j;  //(all_num,1)最多all_num个
        iou_weight_avg += iou * all_score[j];
      }
    }
    int32_t vot_box_num = vt_idx;
    T weight_avg_minx = 0;
    T weight_avg_miny = 0;
    T weight_avg_maxx = 0;
    T weight_avg_maxy = 0;
    T sum_score = 0;
    T genelized_avg_score = 0;
    FOR_RANGE(int32_t, k, 0, vot_box_num) {
      int32_t offset = vot_boxes_inds[k];
      weight_avg_minx += all_box[offset * 4 + 0] * all_score[offset];
      weight_avg_miny += all_box[offset * 4 + 1] * all_score[offset];
      weight_avg_maxx += all_box[offset * 4 + 2] * all_score[offset];
      weight_avg_maxy += all_box[offset * 4 + 3] * all_score[offset];
      T score = all_score[offset];
      sum_score += score;
      genelized_avg_score += std::pow(score, beta);
    }
    nms_box[i * 4 + 0] = weight_avg_minx;
    nms_box[i * 4 + 1] = weight_avg_miny;
    nms_box[i * 4 + 2] = weight_avg_maxx;
    nms_box[i * 4 + 3] = weight_avg_maxy;
    if (score_method == "ID") {
      ;  // do nothing
    } else if (score_method == "TEMP_AVG") {
      TODO();
    } else if (score_method == "AVG") {
      T avg_score = sum_score / vot_box_num;
      nms_score[i] = avg_score;
    } else if (score_method == "IOU_AVG") {
      nms_score[i] = iou_weight_avg;
    } else if (score_method == "GENERALIZED_AVG") {
      genelized_avg_score = genelized_avg_score / vot_box_num;
      genelized_avg_score = pow(genelized_avg_score, 1.0 / beta);
      nms_score[i] = genelized_avg_score;
    } else if (score_method == "QUASI_SUM") {
      T quasi_sum_score = sum_score / pow(static_cast<float>(vot_box_num), beta);
      nms_score[i] = quasi_sum_score;
    } else {
      UNIMPLEMENTED();
    }
  }
}

template<typename T>
T FindScoreThresh(const T* scores_ptr, const int32_t num, const int32_t after_filter_num,
                         int32_t* sorted_score_index_ptr) {
  SortScoreIndex(num, scores_ptr, sorted_score_index_ptr);
  int32_t threshidx = sorted_score_index_ptr[after_filter_num - 1];
  return scores_ptr[threshidx];
}

}  // namespace

template<typename T>
void ResultsWithNmsKernel<T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* boxes_blob = BnInOp2Blob("boxes");    //(im,roi,4*cls)
  const Blob* scores_blob = BnInOp2Blob("scores");  //(im,roi,cls)
  Blob* cls_boxes_blob =
      BnInOp2Blob("cls_boxes");  //(im,cls_num,roi*5)  
  Blob* out_boxes_blob = BnInOp2Blob("out_boxes");    //(im,roi*cls_num,4)
  Blob* out_scores_blob = BnInOp2Blob("out_scores");  //(im,roi*cls_num,1)
  // tmp blob
  Blob* sorted_score_slice_blob = BnInOp2Blob("sorted_score_slice");  //(class_num,roi)
  Blob* supressed_index_blob = BnInOp2Blob("supressed_index");        //(roi,1)
  Blob* score_per_class_blob = BnInOp2Blob("score_per_class");        //(class_num,roi)
  Blob* box_per_class_blob = BnInOp2Blob("box_per_class");            //(class_num,roi*4)
  
  Blob* nms_boxes_blob = BnInOp2Blob("nms_boxes");                    //(cls_num,roi,4)
  Blob* nms_scores_blob = BnInOp2Blob("nms_scores");                  //(cls_num,roi,1)

  
  int32_t* supressed_index_ptr = supressed_index_blob->mut_dptr<int32_t>();
  T* score_per_class_ptr = score_per_class_blob->mut_dptr<T>();
  T* box_per_class_ptr = box_per_class_blob->mut_dptr<T>();

  T* nms_boxes_ptr = nms_boxes_blob->mut_dptr<T>();
  T* nms_scores_ptr = nms_scores_blob->mut_dptr<T>();

  const ResultsWithNmsOpConf& conf = this->op_conf().results_with_nms_conf();
 // const int32_t class_num = conf.class_num();  todo get class_num from score_blob
  const int32_t class_num = scores_blob->shape().At(2);
  const int32_t num_batch = boxes_blob->shape().At(0);
  const int32_t box_num = boxes_blob->shape().At(1);

  FOR_RANGE(int32_t, i, 0, num_batch) {
    int32_t score_offset_per_im = i * box_num * class_num;
    int32_t box_offset_per_im = i * box_num * class_num * 4;
    const T* boxes_ptr = boxes_blob->dptr<T>() + box_offset_per_im;
    const T* scores_ptr = scores_blob->dptr<T>() + score_offset_per_im;
    //todo: modify blob to ofrecord
    T* out_boxes_ptr = out_boxes_blob->mut_dptr<T>() + score_element_offset * 4;
    T* out_scores_ptr = out_scores_blob->mut_dptr<T>() + score_element_offset;
    T* cls_boxes_ptr = cls_boxes_blob->mut_dptr<T>() + score_element_offset * 5;
    
    int32_t keep_num_per_im = 0;
    FOR_RANGE(int32_t, j, 1, class_num) {  // just foreground
      int32_t box_offset_per_cls = j * box_num * 4;
      int32_t score_offset_per_cls = j * box_num;
      T* score_per_class_ptr = score_per_class_blob->mut_dptr<T>() + score_offset_per_cls;   //(cls_num,roi，1)     
      T* box_per_class_ptr = box_per_class_blob->mut_dptr<T>() + box_offset_per_cls;  //(cls_num,roi，4)
      int32_t* sorted_score_slice_ptr = sorted_score_slice_blob->mut_dptr<int32_t>() + score_offset_per_cls;//(cls_num,roi)
      // get score>=thresh boxes and score
      //todo: def a function for this
      int32_t keep_num = 0;
      FOR_RANGE(int32_t, k, 0, box_num) {
        int32_t boxes_offset = k * class_num * 4 + j * 4;
        if (scores_ptr[boxes_offset / 4] > conf.score_thresh()) {
          score_per_class_ptr[keep_num] = scores_ptr[boxes_offset / 4];
          box_per_class_ptr[keep_num * 4 + 0] = boxes_ptr[boxes_offset + 0];
          box_per_class_ptr[keep_num * 4 + 1] = boxes_ptr[boxes_offset + 1];
          box_per_class_ptr[keep_num * 4 + 2] = boxes_ptr[boxes_offset + 2];
          box_per_class_ptr[keep_num * 4 + 3] = boxes_ptr[boxes_offset + 3];
          keep_num++;
        }
      }
      // apply nms
      FasterRcnnUtil<T>::SortByScore(keep_num, score_per_class_ptr,
                                    sorted_score_slice_ptr);
      int32_t nms_keep_num = Nms(box_per_class_ptr, conf.nms_threshold(), sorted_score_index_ptr,
                                 keep_num, supressed_index_ptr);
      int32_t nms_keep_num = FasterRcnnUtil<T>::Nms(box_per_class_ptr, const int32_t* sorted_score_slice_ptr,
                               const int32_t pre_nms_top_n, const int32_t post_nms_top_n,
                               const float nms_threshold, int32_t* area_ptr,
                               int32_t* post_nms_slice_ptr) 
      FOR_RANGE(int32_t, k, 0, nms_keep_num) {
        int32_t boxes_offset = sorted_score_index_ptr[k] * 4;
        nms_boxes_ptr[k * 4 + 0] = box_per_class_ptr[boxes_offset + 0];
        nms_boxes_ptr[k * 4 + 1] = box_per_class_ptr[boxes_offset + 1];
        nms_boxes_ptr[k * 4 + 2] = box_per_class_ptr[boxes_offset + 2];
        nms_boxes_ptr[k * 4 + 3] = box_per_class_ptr[boxes_offset + 3];
        nms_scores_ptr[k] = score_per_class_ptr[boxes_offset / 4];
      }
      // apply vote
      if (conf.bbox_vote_enabled()) {
        BboxVoting(j, nms_keep_num, keep_num, conf.bbox_vote(), BnInOp2Blob);
      }
      keep_num_per_im += nms_keep_num;
    }  // end of cls_num forange

    // Limit to max_per_image detections
    int32_t image_thresh = 0;
    if (conf.detections_per_im() > 0) {
      if (keep_num_per_im > conf.detections_per_im()) {
        image_thresh = FindScoreThresh(nms_scores_ptr, keep_num_per_im, conf.detections_per_im(),
                                        sorted_score_index_ptr);  // find the thresh to filter the
                                                                  // num_per_im to detections_per_im
      }
    }
    // out
    int32_t out_index = 0;
    FOR_RANGE(int32_t, j, 1, class_num) {
      int32_t cls_boxes_offset = j * box_num * 4;
      int32_t cls_boxes_index = 0;
      FOR_RANGE(int32_t, k, 0, box_num) {
        if (nms_scores_ptr[j] >= image_thresh) {
          out_boxes_ptr[out_index * 4 + 0] = nms_boxes_ptr[j * 4 + 0];
          out_boxes_ptr[out_index * 4 + 1] = nms_boxes_ptr[j * 4 + 1];
          out_boxes_ptr[out_index * 4 + 2] = nms_boxes_ptr[j * 4 + 2];
          out_boxes_ptr[out_index * 4 + 3] = nms_boxes_ptr[j * 4 + 3];
          out_scores_ptr[out_index] = nms_scores_ptr[j];
          out_index++;
          cls_boxes_ptr[cls_boxes_offset + cls_boxes_index * 5 + 0] = nms_boxes_ptr[j * 4 + 0];
          cls_boxes_ptr[cls_boxes_offset + cls_boxes_index * 5 + 1] = nms_boxes_ptr[j * 4 + 1];
          cls_boxes_ptr[cls_boxes_offset + cls_boxes_index * 5 + 2] = nms_boxes_ptr[j * 4 + 2];
          cls_boxes_ptr[cls_boxes_offset + cls_boxes_index * 5 + 3] = nms_boxes_ptr[j * 4 + 3];
          cls_boxes_ptr[cls_boxes_offset + cls_boxes_index * 5 + 4] = nms_scores_ptr[j];
          cls_boxes_index++;
        }
      }
    }
  }
}

ADD_CPU_DEFAULT_KERNEL_CREATOR(OperatorConf::kResultsWithNmsConf, ResultsWithNmsKernel,
                               ARITHMETIC_DATA_TYPE_SEQ);

}  // namespace oneflow
