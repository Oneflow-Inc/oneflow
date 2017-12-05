#include "oneflow/core/kernel/boxing_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/operator/op_conf.pb.h"

namespace oneflow {

template<typename T>
void BoxingKernel<T>::GetSumFromSrcBlobsToDstBlob(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob,
    const std::vector<std::string>& src_bns, const std::string& dst_bn) const {
  Blob* dst_blob = BnInOp2Blob(dst_bn);
  Blob* fst_src_blob = BnInOp2Blob(src_bns.front());
  Memcpy<DeviceType::kCPU>(ctx.device_ctx, dst_blob->mut_dptr(),
                           fst_src_blob->dptr(),
                           dst_blob->ByteSizeOfDataContentField(),
                           cudaMemcpyKind::cudaMemcpyHostToHost);
  for (size_t i = 1; i != src_bns.size(); ++i) {
    Blob* src_blob_i = BnInOp2Blob("in_" + std::to_string(i));
    KernelUtil<DeviceType::kCPU, T>::BlasAxpy(
        ctx.device_ctx, dst_blob->shape().elem_cnt(), 1.0, src_blob_i->dptr(),
        1, dst_blob->mut_dptr(), 1);
  }
}

template<typename T>
void BoxingKernel<T>::BoxingCopy(const KernelCtx& ctx, bool is_data_id,
                                 Blob* src_blob, Blob* dst_blob,
                                 const int64_t src_offset,
                                 const int64_t dst_offset, size_t copy_size,
                                 bool need_swap) const {
  if (is_data_id) {
    Memcpy<DeviceType::kCPU>(ctx.device_ctx,
                             dst_blob->mut_data_id() + dst_offset,
                             src_blob->data_id() + src_offset, copy_size,
                             cudaMemcpyKind::cudaMemcpyHostToHost);
  } else if (need_swap) {
    Memcpy<DeviceType::kCPU>(ctx.device_ctx,
                             src_blob->mut_dptr<char>() + src_offset,
                             dst_blob->dptr<char>() + dst_offset, copy_size,
                             cudaMemcpyKind::cudaMemcpyHostToHost);
  } else {
    Memcpy<DeviceType::kCPU>(ctx.device_ctx,
                             dst_blob->mut_dptr<char>() + dst_offset,
                             src_blob->dptr<char>() + src_offset, copy_size,
                             cudaMemcpyKind::cudaMemcpyHostToHost);
  }
}

template<typename T>
void BoxingKernel<T>::CopyDataId(const KernelCtx& ctx,
                                 std::vector<Blob*>& src_blobs,
                                 std::vector<Blob*>& dst_blobs,
                                 const int32_t src_concat_axis,
                                 const int32_t dst_split_axis) const {
  size_t data_id_size = JobDesc::Singleton()->SizeOfOneDataId();
  if (src_concat_axis == 0) {
    int64_t src_idx = 0;
    int64_t dst_idx = 0;
    int64_t src_offset = 0;
    int64_t dst_offset = 0;
    while (src_idx < src_blobs.size() && dst_idx < dst_blobs.size()) {
      int64_t src_cap = src_blobs.at(src_idx)->shape().At(0);
      int64_t dst_cap = dst_blobs.at(dst_idx)->shape().At(0);
      int64_t q = std::min(src_cap - src_offset, dst_cap - dst_offset);

      BoxingCopy(ctx, true, src_blobs.at(src_idx), dst_blobs.at(dst_idx),
                 src_offset * data_id_size, dst_offset * data_id_size,
                 q * data_id_size, false);

      src_offset += q;
      if (src_offset == src_cap) {
        src_offset = 0;
        ++src_idx;
      }

      dst_offset += q;
      if (dst_offset == dst_cap) {
        dst_offset = 0;
        ++dst_idx;
      }
    }
  } else if (src_concat_axis > 0) {
    BoxingCopy(ctx, true, src_blobs.at(0), dst_blobs.at(0), 0, 0,
               src_blobs.at(0)->ByteSizeOfDataIdField(), false);
  } else {
    UNEXPECTED_RUN();
  }

  if (dst_split_axis > 0) {
    // add copy rules from first dst blob to all dst blobs
    for (size_t i = 1; i < dst_blobs.size(); ++i) {
      BoxingCopy(ctx, true, dst_blobs.at(0), dst_blobs.at(i), 0, 0,
                 dst_blobs.at(0)->ByteSizeOfDataIdField(), false);
    }
  }
}

template<typename T>
void BoxingKernel<T>::DoUnequalAxisCopy(
    const KernelCtx& ctx, std::vector<Blob*>& src_blobs,
    std::vector<Blob*>& dst_blobs, const int32_t src_axis,
    const int32_t dst_axis, const BoxingInfo& src_info,
    const BoxingInfo& dst_info, bool need_swap) const {
  for (size_t src_idx = 0; src_idx != src_blobs.size(); ++src_idx) {
    for (size_t seg_idx = 0; seg_idx != src_info.total_seg_num(); ++seg_idx) {
      int64_t src_seg_offset = seg_idx * src_info.size_of_per_seg()
                               + src_info.offset_of_subseg(src_idx);
      int64_t dst_segs_in_src_seg =
          src_info.size_of_per_seg() / dst_info.size_of_per_seg();
      for (size_t dst_seg_idx = 0; dst_seg_idx != dst_segs_in_src_seg;
           ++dst_seg_idx) {
        int64_t dst_seg_offset = 0;
        for (size_t dst_idx = 0; dst_idx != dst_blobs.size(); ++dst_idx) {
          int64_t dst_seg_start = src_seg_offset / dst_info.size_of_per_seg()
                                  * dst_info.size_of_subseg(dst_idx);
          BoxingCopy(ctx, src_blobs.at(src_idx), dst_blobs.at(dst_idx),
                     src_seg_offset * sizeof(T),
                     (dst_seg_start + dst_seg_offset) * sizeof(T),
                     dst_info.size_of_subseg(dst_idx) * sizeof(T), need_swap);
          src_seg_offset += dst_info.size_of_subseg(dst_idx);
          dst_seg_offset += dst_info.size_of_subseg(dst_idx);
        }
      }
      src_seg_offset += src_info.size_of_per_seg();
    }
  }
}

template<typename T>
void BoxingKernel<T>::BoxingCopyForUnequalAxis(const KernelCtx& ctx,
                                               std::vector<Blob*>& src_blobs,
                                               std::vector<Blob*>& dst_blobs,
                                               const int32_t concat_axis,
                                               const int32_t split_axis) const {
  auto kernel_conf = this->kernel_conf().boxing_conf();
  const BoxingInfo& in_info = kernel_conf.in_info();
  const BoxingInfo& out_info = kernel_conf.out_info();
  if (concat_axis > split_axis) {
    DoUnequalAxisCopy(ctx, dst_blobs, src_blobs, split_axis, concat_axis,
                      out_info, in_info, true);
  } else {
    DoUnequalAxisCopy(ctx, src_blobs, dst_blobs, concat_axis, split_axis,
                      in_info, out_info, false);
  }
}

template<typename T>
void BoxingKernel<T>::BoxingCopyForEqualAxis(const KernelCtx& ctx,
                                             std::vector<Blob*>& src_blobs,
                                             std::vector<Blob*>& dst_blobs,
                                             const int32_t axis) const {
  // P.S This routine will be called only once, thus some performance
  // loss seems ok.
  auto kernel_conf = this->kernel_conf().boxing_conf();
  const BoxingInfo& in_info = kernel_conf.in_info();
  const BoxingInfo& out_info = kernel_conf.out_info();
  int64_t src_offset = 0;
  for (size_t src_idx = 0, dst_idx = 0;
       src_idx != src_blobs.size() && dst_idx != dst_blobs.size();) {
    int64_t dst_offset = 0;
    while (dst_offset < out_info.size_of_subseg(dst_idx)) {
      int64_t p = std::min(in_info.size_of_subseg(src_idx) - src_offset,
                           out_info.size_of_subseg(dst_idx) - dst_offset);
      for (size_t i = 0; i != in_info.total_seg_num(); ++i) {
        BoxingCopy(
            ctx, true, src_blobs.at(src_idx), dst_blobs.at(dst_idx),
            (src_offset + i * in_info.size_of_subseg(src_idx)) * sizeof(T),
            (dst_offset + i * out_info.size_of_subseg(dst_idx)) * sizeof(T),
            p * sizeof(T), false);
      }
      src_offset += p;
      dst_offset += p;
      if (src_offset == in_info.size_of_subseg(src_idx)) {
        if (++src_idx == src_blobs.size()) { break; }
        src_offset = 0;
      }
    }
    dst_idx++;
  }
}

template<typename T>
void BoxingKernel<T>::CopyFromSrcBlobs2DstBlobs(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob,
    const std::vector<std::string>& src_bns,
    const std::vector<std::string>& dst_bns, const int32_t src_concat_axis,
    const int32_t dst_split_axis) const {
  std::vector<Blob*> src_blobs;
  std::vector<Blob*> dst_blobs;
  for (const std::string& bn : src_bns) {
    Blob* b = BnInOp2Blob(bn);
    if (b == nullptr) { break; }
    src_blobs.emplace_back(b);
  }
  for (const std::string& bn : dst_bns) {
    Blob* b = BnInOp2Blob(bn);
    if (b == nullptr) { break; }
    dst_blobs.emplace_back(b);
  }
  if (src_blobs.front()->has_data_id()) {
    CopyDataId(ctx, src_blobs, dst_blobs, src_concat_axis, dst_split_axis);
  }

  if (src_concat_axis == dst_split_axis) {
    BoxingCopyForEqualAxis(ctx, src_blobs, dst_blobs, src_concat_axis);
  } else {
    BoxingCopyForUnequalAxis(ctx, src_blobs, dst_blobs, src_concat_axis,
                             dst_split_axis);
  }
}

template<typename T>
void BoxingKernel<T>::CopyFromFirstBlob2OtherBlobs(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob,
    const std::vector<std::string>& obns) const {
  int64_t copy_size = BnInOp2Blob(obns.front())->shape().elem_cnt() * sizeof(T);
  FOR_RANGE(size_t, i, 1, obns.size()) {
    BoxingCopy(ctx, false, BnInOp2Blob(obns.front()), BnInOp2Blob(obns.at(i)),
               0, 0, copy_size, false);
  }
}

template<typename T>
void BoxingKernel<T>::Forward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  auto boxing_conf = op_conf().boxing_conf();
  const KernelConf& kernel_conf = this->kernel_conf();
  if (boxing_conf.in_box_case() == BoxingOpConf::kConcatBox) {
    // concat-box copy rules: copy directly from input to output
    int32_t concat_axis = boxing_conf.concat_box().axis();
    if (boxing_conf.out_box_case() == BoxingOpConf::kSplitBox) {
      CopyFromSrcBlobs2DstBlobs(ctx, BnInOp2Blob, kernel_conf.input_bns(),
                                kernel_conf.output_bns(), concat_axis,
                                boxing_conf.split_box().axis());
    } else if (boxing_conf.out_box_case() == BoxingOpConf::kCloneBox) {
      CopyFromSrcBlobs2DstBlobs(ctx, BnInOp2Blob, kernel_conf.input_bns(),
                                {"out_0"}, concat_axis, 0);
      CopyFromFirstBlob2OtherBlobs(ctx, BnInOp2Blob, kernel_conf.output_bns());
    } else {
      UNEXPECTED_RUN();
    }
  } else if (boxing_conf.in_box_case() == BoxingOpConf::kAddBox) {
    CHECK_EQ(boxing_conf.out_box_case(), BoxingOpConf::kSplitBox);
    GetSumFromSrcBlobsToDstBlob(ctx, BnInOp2Blob, kernel_conf.input_bns(),
                                {"middle"});
    CopyFromSrcBlobs2DstBlobs(ctx, BnInOp2Blob, {"middle"},
                              kernel_conf.output_bns(), 0,
                              boxing_conf.split_box().axis());
  } else {
    UNEXPECTED_RUN();
  }
}

namespace {

Kernel* CreateBoxingKernel(const KernelConf& kernel_conf) {
  static const HashMap<std::string, std::function<Kernel*()>> creators = {
#define BOXING_KERNEL_ENTRY(data_type_pair)       \
  {GetHashKey(OF_PP_PAIR_SECOND(data_type_pair)), \
   []() { return new BoxingKernel<OF_PP_PAIR_FIRST(data_type_pair)>(); }},
      OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(BOXING_KERNEL_ENTRY,
                                       FLOATING_DATA_TYPE_SEQ)};
  return creators.at(GetHashKey(JobDesc::Singleton()->DefaultDataType()))();
}

}  // namespace

}  // namespace oneflow
