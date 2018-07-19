#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/kernel/roi_pooling_kernel.h"

namespace oneflow {

namespace {

template<typename T>
__global__ void RoIPoolForward(const int64_t nthreads, const T* in_dptr, const float spatial_scale,
                               const int64_t channel_num, const int64_t height, const int64_t width,
                               const int64_t roi_num, const int64_t pooled_height,
                               const int64_t pooled_width, const T* rois_dptr, T* out_dptr,
                               int32_t* argmax_dptr) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    const int64_t pooled_area = pooled_width * pooled_height;
    const int64_t w = index % pooled_width;
    const int64_t h = (index / pooled_width) % pooled_height;
    const int64_t c = (index / pooled_area) % channel_num;
    const int64_t r = (index / pooled_area / channel_num) % roi_num;
    const int64_t n = index / pooled_area / channel_num / roi_num;
    const T* offset_rois = rois_dptr + n * roi_num * 4 + r * 4;
    int64_t roi_start_h =
        min(max(static_cast<int64_t>(round(offset_rois[0] * spatial_scale)), 0l), height);
    int64_t roi_start_w =
        min(max(static_cast<int64_t>(round(offset_rois[1] * spatial_scale)), 0l), width);
    int64_t roi_end_h =
        min(max(static_cast<int64_t>(round(offset_rois[2] * spatial_scale)), 0l), height);
    int64_t roi_end_w =
        min(max(static_cast<int64_t>(round(offset_rois[3] * spatial_scale)), 0l), width);
    int64_t roi_width = min(roi_end_w - roi_start_w, 1l);
    int64_t roi_height = min(roi_end_h - roi_start_h, 1l);
    const float bin_height = static_cast<float>(roi_height) / static_cast<float>(pooled_height);
    const float bin_weight = static_cast<float>(roi_width) / static_cast<float>(pooled_width);
    int64_t hstart = floor(static_cast<float>(h) * bin_height);
    int64_t wstart = floor(static_cast<float>(w) * bin_weight);
    int64_t hend = ceil(static_cast<float>(h + 1) * bin_height);
    int64_t wend = ceil(static_cast<float>(w + 1) * bin_weight);
    hstart = min(max(roi_start_h + hstart, 0l), height);
    wstart = min(max(roi_start_w + wstart, 0l), width);
    hend = min(max(roi_start_h + hend, 0l), height);
    wend = min(max(roi_start_w + wend, 0l), width);
    bool is_bin_empty = (hend <= hstart) || (wend <= wstart);
    T maxval = 0;
    int64_t maxidx = -1;
    if (!is_bin_empty) {
      const T* offset_in_dptr = in_dptr + (n * channel_num + c) * height * width;
      FOR_RANGE(int64_t, feat_h, hstart, hend) {
        FOR_RANGE(int64_t, feat_w, wstart, wend) {
          if (offset_in_dptr[feat_h * width + feat_w] > maxval) {
            maxidx = feat_h * width + feat_w;
            maxval = offset_in_dptr[maxidx];
          }
        }
      }
    }
    out_dptr[index] = maxval;
    argmax_dptr[index] = maxidx;
  }
}

}  // namespace

template<typename T>
class RoIPoolingKernelUtil<DeviceType::kGPU, T> final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RoIPoolingKernelUtil);
  RoIPoolingKernelUtil() = delete;

  static void Forward(const KernelCtx& ctx, const RoIPoolingOpConf& conf, const Blob* in_blob,
                      const Blob* rois_blob, Blob* out_blob, Blob* argmax_blob) {
    const int64_t count = out_blob->shape().elem_cnt();
    RoIPoolForward<T><<<BlocksNum4ThreadsNum(count), kCudaThreadsNumPerBlock, 0,
                        ctx.device_ctx->cuda_stream()>>>(
        count, in_blob->dptr<T>(), conf.spatial_scale(), in_blob->shape().At(1),
        in_blob->shape().At(2), in_blob->shape().At(3), out_blob->shape().At(1), conf.pooled_h(),
        conf.pooled_w(), rois_blob->dptr<T>(), out_blob->mut_dptr<T>(),
        argmax_blob->mut_dptr<int32_t>());
  }

  static void Backward(const KernelCtx& ctx, const RoIPoolingOpConf& conf,
                       const Blob* out_diff_blob, const Blob* rois_blob, const Blob* argmax_blob,
                       Blob* in_diff_blob) {}
};

#define INSTANTIATE_ROI_POOLING_KERNEL_UTIL(type_cpp, type_proto) \
  template class RoIPoolingKernelUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_ROI_POOLING_KERNEL_UTIL,
                     OF_PP_MAKE_TUPLE_SEQ(float, DataType::kFloat))

}  // namespace oneflow
