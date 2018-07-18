#include "oneflow/core/kernel/proposal_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

namespace {

void GenerateAnchors(const ProposalOpConf& conf, Blob* anchors_blob) {
  // anchors_blob shape (H * W * A, 4)
  int32_t* anchors_dptr = anchors_blob->mut_dptr<int32_t>();
  int32_t height = anchors_blob->shape().At(1);
  int32_t width = anchors_blob->shape().At(2);
  int32_t scales_size = conf.anchor_scales_size();
  int32_t ratios_size = conf.aspect_ratios_size();
  int32_t num_anchors = scales_size * ratios_size;
  int32_t fm_stride = conf.feature_map_stride();
  float base_ctr = 0.5 * (fm_stride - 1);

  std::vector<int32_t> base_anchors(num_anchors * 4);
  FOR_RANGE(int32_t, i, 0, scales_size) {
    FOR_RANGE(int32_t, j, 0, ratios_size) {
      int32_t ws = width * conf.anchor_scales(i);
      int32_t hs = height * conf.anchor_scales(i);
      float wr = std::sqrt(hs * ws / conf.aspect_ratios(j));
      float hr = wr * conf.aspect_ratios(j);
      base_anchors[i * ratios_size * 4 + j * 4] = base_ctr - 0.5 * (wr - 1);
      base_anchors[i * ratios_size * 4 + j * 4 + 1] = base_ctr - 0.5 * (hr - 1);
      base_anchors[i * ratios_size * 4 + j * 4 + 2] = base_ctr + 0.5 * (wr - 1);
      base_anchors[i * ratios_size * 4 + j * 4 + 3] = base_ctr + 0.5 * (hr - 1);
    }
  }

  FOR_RANGE(int32_t, h, 0, height) {
    FOR_RANGE(int32_t, w, 0, width) {
      int32_t* anchors = anchors_dptr + (h * width + w) * num_anchors * 4;
      FOR_RANGE(int32_t, i, 0, base_anchors.size()) {
        if (i % 2 == 0) {
          *(anchors + i) = base_anchors[i] + w * fm_stride;
        } else {
          *(anchors + i) = base_anchors[i] + h * fm_stride;
        }
      }
    }
  }
}

}  // namespace

template<typename T>
void ProposalKernel<T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {}

template<typename T>
void ProposalKernel<T>::InitConstBufBlobs(
    DeviceCtx* ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  GenerateAnchors(op_conf().proposal_conf(), BnInOp2Blob("anchors"));
}

ADD_CPU_DEFAULT_KERNEL_CREATOR(OperatorConf::kProposalConf, ProposalKernel,
                               ARITHMETIC_DATA_TYPE_SEQ);

}  // namespace oneflow
