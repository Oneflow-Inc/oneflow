#include "oneflow/core/kernel/roi_pooling_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void RoIPoolingKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  const Blob* rois_blob = BnInOp2Blob("rois");
  Blob* out_blob = BnInOp2Blob("out");
  Blob* argmax_blob = BnInOp2Blob("argmax");
  Memset<device_type>(ctx.device_ctx, argmax_blob->mut_dptr<int32_t>(), -1,
                      argmax_blob->ByteSizeOfDataContentField());
  Memset<device_type>(ctx.device_ctx, out_blob->mut_dptr<T>(), 0,
                      out_blob->ByteSizeOfDataContentField());
  RoIPoolingKernelUtil<device_type, T>::Forward(ctx, this->op_conf().roi_pooling_conf(), in_blob,
                                                rois_blob, out_blob, argmax_blob);
}

template<DeviceType device_type, typename T>
void RoIPoolingKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* in_diff_blob = BnInOp2Blob("in_diff");
  if (in_diff_blob == nullptr) { return; }
  Memset<device_type>(ctx.device_ctx, in_diff_blob->mut_dptr<T>(), 0,
                      in_diff_blob->ByteSizeOfDataContentField());
  const Blob* out_diff_blob = BnInOp2Blob("out_diff");
  const Blob* rois_blob = BnInOp2Blob("rois");
  const Blob* argmax_blob = BnInOp2Blob("argmax");
  RoIPoolingKernelUtil<device_type, T>::Backward(
      ctx, this->op_conf().roi_pooling_conf(), out_diff_blob, rois_blob, argmax_blob, in_diff_blob);
}

template<typename T>
class RoIPoolingKernelUtil<DeviceType::kCPU, T> final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RoIPoolingKernelUtil);
  RoIPoolingKernelUtil() = delete;

  static void Forward(const KernelCtx& ctx, const RoIPoolingOpConf& conf, const Blob* in_blob,
                      const Blob* rois_blob, Blob* out_blob, Blob* argmax_blob) {
    const T* in_dptr = in_blob->dptr<T>();
    const float spatial_scale = conf.spatial_scale();
    const int64_t height = in_blob->shape().At(2);
    const int64_t width = in_blob->shape().At(3);
    const int64_t pooled_height = conf.pooled_h();
    const int64_t pooled_width = conf.pooled_w();
    const int64_t pooled_area = out_blob->shape().Count(3);  // pooled_h * pooled_w
    const int64_t roi_num = rois_blob->shape().At(1);
    const int64_t roi_size = rois_blob->shape().Count(1);  // roi_num * 4
    const int64_t out_size =
        out_blob->shape().Count(1);  // roi_num * channel_num * pooled_h * pooled_w
    FOR_RANGE(int64_t, n, 0, in_blob->shape().At(0)) {
      const T* rois_dptr = rois_blob->dptr<T>() + roi_size * n;
      int64_t n_out_size = out_size * n;
      T* out_dptr = out_blob->mut_dptr<T>() + n_out_size;
      int32_t* argmax_dptr = argmax_blob->mut_dptr<int32_t>() + n_out_size;
      FOR_RANGE(int64_t, r, 0, roi_num) {
        // stay within feature map
        int64_t roi_start_w = std::min(
            std::max(static_cast<int64_t>(round(rois_dptr[r * 4] * spatial_scale)), 0l), height);
        int64_t roi_start_h = std::min(
            std::max(static_cast<int64_t>(round(rois_dptr[r * 4 + 1] * spatial_scale)), 0l), width);
        int64_t roi_end_w = std::min(
            std::max(static_cast<int64_t>(round(rois_dptr[r * 4 + 2] * spatial_scale)), 0l),
            height);
        int64_t roi_end_h = std::min(
            std::max(static_cast<int64_t>(round(rois_dptr[r * 4 + 3] * spatial_scale)), 0l), width);
        // no smaller than 1 * 1
        int64_t roi_height = std::max<int64_t>(roi_end_h - roi_start_h + 1, 1);
        int64_t roi_width = std::max<int64_t>(roi_end_w - roi_start_w + 1, 1);
        const float bin_height = static_cast<float>(roi_height) / static_cast<float>(pooled_height);
        const float bin_width = static_cast<float>(roi_width) / static_cast<float>(pooled_width);
        FOR_RANGE(int64_t, c, 0, in_blob->shape().At(1)) {
          FOR_RANGE(int64_t, h, 0, pooled_height) {
            FOR_RANGE(int64_t, w, 0, pooled_width) {
              int64_t hstart = floor(static_cast<float>(h) * bin_height);
              int64_t wstart = floor(static_cast<float>(w) * bin_width);
              int64_t hend = ceil(static_cast<float>(h + 1) * bin_height);
              int64_t wend = ceil(static_cast<float>(w + 1) * bin_width);
              hstart = std::min(std::max(roi_start_h + hstart, 0l), height);
              wstart = std::min(std::max(roi_start_w + wstart, 0l), width);
              hend = std::min(std::max(roi_start_h + hend, 0l), height);
              wend = std::min(std::max(roi_start_w + wend, 0l), width);
              int64_t out_pos =
                  r * out_blob->shape().Count(2) + c * pooled_area + h * pooled_width + w;
              bool is_bin_empty = (hend <= hstart) || (wend <= wstart);
              out_dptr[out_pos] = is_bin_empty ? 0 : -FLT_MAX;
              argmax_dptr[out_pos] = -1;
              if (!is_bin_empty) {
                const T* offset_in_dptr = in_dptr + c * in_blob->shape().Count(2);
                FOR_RANGE(int64_t, feat_h, hstart, hend) {
                  FOR_RANGE(int64_t, feat_w, wstart, wend) {
                    int32_t in_pos = feat_h * in_blob->shape().Count(3) + feat_w;
                    if (offset_in_dptr[in_pos] > out_dptr[out_pos]) {
                      out_dptr[out_pos] = offset_in_dptr[in_pos];
                      argmax_dptr[out_pos] = in_pos;
                    }
                  }
                }
              }
            }
          }
        }  // channels
      }    // rois
      in_dptr += in_blob->shape().Count(1);
    }  // batch
  }

  static void Backward(const KernelCtx& ctx, const RoIPoolingOpConf& conf,
                       const Blob* out_diff_blob, const Blob* rois_blob, const Blob* argmax_blob,
                       Blob* in_diff_blob) {}
};

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kRoiPoolingConf, RoIPoolingKernel,
                           OF_PP_MAKE_TUPLE_SEQ(float, DataType::kFloat));

}  // namespace oneflow
