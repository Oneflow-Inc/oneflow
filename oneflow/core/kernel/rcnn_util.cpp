#include "oneflow/core/kernel/rcnn_util.h"

namespace oneflow {

template<typename T>
float RcnnUtil<T>::BBoxArea(const T* box_dptr) {
  return (box_dptr[2] - box_dptr[0] + 1) * (box_dptr[3] - box_dptr[1] + 1);
}

template<typename T>
float RcnnUtil<T>::InterOverUnion(const T* box1_dptr, const T* box2_dptr) {
  float boxIou = 0;
  int32_t iw = std::min(box1_dptr[2], box2_dptr[2]) - std::max(box1_dptr[0], box2_dptr[0]) + 1;
  if (iw > 0) {
    int32_t ih = std::min(box1_dptr[3], box2_dptr[3]) - std::max(box1_dptr[1], box2_dptr[1]) + 1;
    if (ih > 0) {
      float ua = BBoxArea(box1_dptr) + BBoxArea(box2_dptr) - iw * ih;
      boxIou = iw * ih / ua;
    }
  }
  return boxIou;
}

template<typename T>
void RcnnUtil<T>::BboxOverlaps(const T* rois, const int32_t roi_num, const T* gt_boxes,
                               const int32_t gt_num, const int32_t gt_max_num, T* overlaps) {
  FOR_RANGE(int32_t, i, 0, roi_num) {
    FOR_RANGE(int32_t, j, 0, gt_num) {
      overlaps[i * gt_max_num + j] = InterOverUnion(rois + i * 4, gt_boxes + j * 5);
    }
  }
}

template<typename T>
void RcnnUtil<T>::OverlapRowArgMax7Max(const T* overlaps, const int32_t roi_num,
                                       const int32_t gt_num, const int32_t gt_max_num,
                                       T* argmax_dptr, T* max_dptr) {
  FOR_RANGE(int32_t, i, 0, roi_num) {
    FOR_RANGE(int32_t, j, 0, gt_num) {
      if (max_dptr[i] < overlaps[i * gt_max_num + j]) {
        max_dptr[i] = overlaps[i * gt_max_num + j];
        argmax_dptr[i] = j;
      }
    }
  }
}

template<typename T>
void RcnnUtil<T>::OverlapColArgMax7Max(const T* overlaps, const int32_t roi_num,
                                       const int32_t gt_num, const int32_t gt_max_num,
                                       T* argmax_dptr, T* max_dptr) {
  FOR_RANGE(int32_t, i, 0, gt_num) {
    FOR_RANGE(int32_t, j, 0, roi_num) {
      if (max_dptr[i] < overlaps[j * gt_max_num + i]) {
        max_dptr[i] = overlaps[j * gt_max_num + i];
        argmax_dptr[i] = j;
      }
    }
  }
}

template<typename T>
void RcnnUtil<T>::SampleChoice(T* rois_ptr, const int32_t num, T* sample_ptr,
                               const int32_t sample_num) {
  if (num == sample_num) {
    FOR_RANGE(int32_t, i, 0, num) { sample_ptr[i] = rois_ptr[i]; }
    return;
  }
  // get not repeating sample
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, num - 1);
  for (int i = 0; i < sample_num; i++) {
    int32_t randn = dis(gen);
    if (rois_ptr[randn] == -1) {
      i--;
    } else {
      sample_ptr[i] = rois_ptr[randn];
      rois_ptr[randn] = -1;
    }
  }
}

#define INSTANTIATE_KERNEL_UTIL(type_cpp, type_proto) template struct RcnnUtil<type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_KERNEL_UTIL, ARITHMETIC_DATA_TYPE_SEQ);

}  // namespace oneflow
