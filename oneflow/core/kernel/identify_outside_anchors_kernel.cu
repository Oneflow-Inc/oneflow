#include "oneflow/core/kernel/identify_outside_anchors_kernel_util.h"

namespace oneflow {

namespace {

template<typename T>
__global__ void IdentifyOutsideAnchorsGpu(const T* anchors, int32_t num_anchors,
                                          const int32_t* image_size, float tolerance,
                                          int8_t* identifications) {
  CUDA_1D_KERNEL_LOOP(i, num_anchors) {
    const T* cur_anchor_ptr = anchors + i * 4;
    if (cur_anchor_ptr[0] < -tolerance || cur_anchor_ptr[1] < -tolerance
        || cur_anchor_ptr[2] >= image_size[1] + tolerance
        || cur_anchor_ptr[3] >= image_size[0] + tolerance) {
      identifications[i] = 1;
    }
  }
}

}  // namespace

template<typename T>
struct IdentifyOutsideAnchorsUtil<DeviceType::kGPU, T> {
  static void IdentifyOutsideAnchors(DeviceCtx* ctx, const Blob* anchors_blob,
                                     const Blob* image_size_blob, Blob* identification_blob,
                                     float tolerance) {
    const T* anchors_dptr = anchors_blob->dptr<T>();
    int32_t num_anchors = static_cast<int32_t>(anchors_blob->shape().At(0));
    const int32_t* image_size_dptr = image_size_blob->dptr<int32_t>();
    int8_t* identification_dptr = identification_blob->mut_dptr<int8_t>();
    Memset<DeviceType::kGPU>(ctx, identification_dptr, 0,
                             identification_blob->ByteSizeOfBlobBody());
    IdentifyOutsideAnchorsGpu<<<BlocksNum4ThreadsNum(anchors_blob->shape().At(0)),
                                kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
        anchors_dptr, num_anchors, image_size_dptr, tolerance, identification_dptr);
  }
};

#define INSTANTIATE_IDENTIFY_OUTSIDE_ANCHORS_UTIL(T, TProto) \
  template struct IdentifyOutsideAnchorsUtil<DeviceType::kGPU, T>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_IDENTIFY_OUTSIDE_ANCHORS_UTIL, FLOATING_DATA_TYPE_SEQ)
#undef INSTANTIATE_IDENTIFY_OUTSIDE_ANCHORS_UTIL

}  // namespace oneflow