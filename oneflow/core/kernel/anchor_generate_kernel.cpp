#include "oneflow/core/kernel/kernel.h"
#include <cfenv>

namespace oneflow {

template<typename T>
class AnchorGenerateKernel final : public KernelIf<DeviceType::kCPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AnchorGenerateKernel);
  AnchorGenerateKernel() = default;
  ~AnchorGenerateKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    const AnchorGenerateOpConf& conf = op_conf().anchor_generate_conf();
    const Blob* images_blob = BnInOp2Blob("images");
    const int64_t batch_height = images_blob->shape().At(1);
    const int64_t batch_width = images_blob->shape().At(2);

    Blob* anchors = BnInOp2Blob("anchors");
    Memset<DeviceType::kCPU>(ctx.device_ctx, anchors->mut_dptr<T>(), 0,
                             anchors->ByteSizeOfBlobBody());
    const float fm_stride = static_cast<float>(conf.feature_map_stride());
    const int32_t feature_map_height = std::ceil(static_cast<float>(batch_height) / fm_stride);
    const int32_t feature_map_width = std::ceil(static_cast<float>(batch_width) / fm_stride);
    auto scales_vec = PbRf2StdVec(conf.anchor_scales());
    auto ratios_vec = PbRf2StdVec(conf.aspect_ratios());
    const size_t num_anchors = GenerateAnchors(fm_stride, feature_map_height, feature_map_width,
                                               scales_vec, ratios_vec, anchors->mut_dptr<T>());
    CHECK_LE(num_anchors, anchors->static_shape().At(0));
  }

  size_t GenerateAnchors(float feature_map_stride, int32_t feature_map_height,
                         int32_t feature_map_width, const std::vector<float>& scales_vec,
                         const std::vector<float>& ratios_vec, T* anchors_ptr) const {
    const float base_ctr = 0.5 * (feature_map_stride - 1);
    const size_t num_anchors = scales_vec.size() * ratios_vec.size();
    std::vector<T> base_anchors_vec(num_anchors * 4);

    int save_round_way = std::fegetround();
    CHECK_EQ(std::fesetround(FE_TONEAREST), 0);
    // scale first, ratio last
    FOR_RANGE(int32_t, i, 0, ratios_vec.size()) {
      const int32_t wr =
          std::nearbyint(std::sqrt(feature_map_stride * feature_map_stride / ratios_vec.at(i)));
      const int32_t hr = std::nearbyint(wr * ratios_vec.at(i));
      FOR_RANGE(int32_t, j, 0, scales_vec.size()) {
        const float scale = scales_vec.at(j) / feature_map_stride;
        const int32_t ws = wr * scale;
        const int32_t hs = hr * scale;
        const int32_t cur_anchor_idx = i * scales_vec.size() + j;
        base_anchors_vec[cur_anchor_idx * 4 + 0] = base_ctr - 0.5 * (ws - 1);  // x1
        base_anchors_vec[cur_anchor_idx * 4 + 1] = base_ctr - 0.5 * (hs - 1);  // y1
        base_anchors_vec[cur_anchor_idx * 4 + 2] = base_ctr + 0.5 * (ws - 1);  // x2
        base_anchors_vec[cur_anchor_idx * 4 + 3] = base_ctr + 0.5 * (hs - 1);  // y2
      }
    }
    std::fesetround(save_round_way);

    FOR_RANGE(int32_t, h, 0, feature_map_height) {
      FOR_RANGE(int32_t, w, 0, feature_map_width) {
        auto* cur_anchor_ptr = anchors_ptr + (h * feature_map_width + w) * num_anchors * 4;
        FOR_RANGE(int32_t, i, 0, num_anchors) {
          cur_anchor_ptr[i * 4 + 0] = base_anchors_vec[i * 4 + 0] + w * feature_map_stride;  // x1
          cur_anchor_ptr[i * 4 + 1] = base_anchors_vec[i * 4 + 1] + h * feature_map_stride;  // y1
          cur_anchor_ptr[i * 4 + 2] = base_anchors_vec[i * 4 + 2] + w * feature_map_stride;  // x2
          cur_anchor_ptr[i * 4 + 3] = base_anchors_vec[i * 4 + 3] + h * feature_map_stride;  // y2
        }
      }
    }
    return num_anchors * feature_map_height * feature_map_width;
  }
};

#define REGISTER_ANCHOR_GENERATE_KERNEL(dtype)                                               \
  REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kAnchorGenerateConf, DeviceType::kCPU, \
                                        dtype, AnchorGenerateKernel<dtype>)

REGISTER_ANCHOR_GENERATE_KERNEL(float);
REGISTER_ANCHOR_GENERATE_KERNEL(double);

}  // namespace oneflow
