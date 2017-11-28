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
  Memcpy<DeviceType::kCPU>(ctx.device_ctx, dst_blob->mut_dptr<T>(),
                           fst_src_blob->dptr<T>(),
                           dst_blob->ByteSizeOfDataContentField(),
                           cudaMemcpyKind::cudaMemcpyHostToHost);
  for (size_t i = 1; i != src_bns.size(); ++i) {
    Blob* src_blob_i = BnInOp2Blob("in_" + std::to_string(i));
    KernelUtil<DeviceType::kCPU, T>::BlasAxpy(
        ctx.device_ctx, dst_blob->shape().elem_cnt(), static_cast<T>(1.0),
        src_blob_i->dptr<T>(), 1, dst_blob->mut_dptr<T>(), 1);
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

  if (dst_split_axis == 1) {
    // add copy rules from first dst blob to all dst blobs
    for (size_t i = 1; i < dst_blobs.size(); ++i) {
      BoxingCopy(ctx, true, dst_blobs.at(0), dst_blobs.at(i), 0, 0,
                 dst_blobs.at(0)->ByteSizeOfDataIdField(), false);
    }
  }
}

template<typename T>
void BoxingKernel<T>::InferCopyRulesFromUnequalAxis(
    const KernelCtx& ctx, std::vector<Blob*>& src_blobs,
    std::vector<Blob*>& dst_blobs, const int32_t concat_axis,
    const int32_t split_axis) const {
  bool need_swap = false;
  if (concat_axis > split_axis) { need_swap = true; }
  int64_t src_seg_cnt =
      (concat_axis == 0) ? 1
                         : (src_blobs.front()->shape().Count(0, concat_axis));
  std::vector<int64_t> src_offset_in_seg(src_blobs.size());
  int64_t src_total_seg_size = 0;
  for (size_t i = 0; i != src_blobs.size(); ++i) {
    src_offset_in_seg = src_total_seg_size;
    src_total_seg_size += src_blobs.at(i)->shape().Count(concat_axis);
  }
  int64_t dst_total_seg_size = 0;
  for (Blob* dst_blob : dst_blobs) {
    dst_total_seg_size += dst_blob->shape().Count(concat_axis);
  }
  for (size_t src_idx = 0; src_idx != src_blobs.size(); ++src_idx) {
    int64_t src_seg_offset =
        src_idx * src_total_seg_size + src_offset_in_seg.at(src_idx);
    for (size_t seg_idx = 0; seg_idx != src_seg_cnt; ++seg_idx) {
      for (size_t dst_seg_idx = 0;
           dst_seg_idx != src_total_seg_size / dst_total_seg_size;
           ++dst_seg_idx) {
        int64_t dst_seg_offset = 0;
        for (size_t dst_idx = 0; dst_idx != dst_blobs.size(); ++dst_idx) {
          int64_t dst_seg_start =
              src_seg_offset / dst_total_seg_size
              * dst_blobs.at(dst_idx)->shape().Count(split_axis);
          BoxingCopy(ctx, src_blobs.at(src_idx), dst_blobs.at(dst_idx),
                     src_seg_offset, dst_seg_start + dst_seg_offset, need_swap);
          src_seg_offset += dst_blobs.at(dst_idx)->shape().Count(split_axis);
          dst_seg_offset += dst_blobs.at(dst_idx)->shape().Count(split_axis);
        }
      }
      src_seg_offset += src_total_seg_size;
    }
  }
}

template<typename T>
void BoxingKernel<T>::InferCopyRulesFromConcatDim(
    const KernelCtx& ctx,
    const std::map<const std::string*, int64_t>& src_bn2concat_dim,
    const std::map<const std::string*, int64_t>& dst_bn2concat_dim,
    const int64_t seg_cnt, const int64_t concat_dim_sz,
    const int32_t concat_axis) const {
  int64_t src_offset = 0;
  for (auto src_iter = src_bn2concat_dim.begin(),
            dst_iter = dst_bn2concat_dim.begin();
       src_iter != src_bn2concat_dim.end()
       && dst_iter != dst_bn2concat_dim.end();) {
    int64_t dst_offset = 0;
    while (dst_offset < dst_iter->second) {
      int64_t p = std::min(src_iter->second - src_offset,
                           dst_iter->second - dst_offset);
      for (size_t i = 0; i < seg_cnt; ++i) {
        BoxingCopy(
            ctx, false, *src_iter->first, *dst_iter->first,
            (src_offset + i * src_iter->second) * concat_dim_sz * sizeof(T),
            (dst_offset + i * dst_iter->second) * concat_dim_sz * sizeof(T),
            p * concat_dim_sz * sizeof(T), false);
      }
      src_offset += p;
      dst_offset += p;
      if (src_offset == src_iter->second) {
        if (++src_iter == src_bn2concat_dim.end()) { break; }
        src_offset = 0;
      }
    }  // while current dst box is not full
    ++dst_iter;
  }
}

template<typename T>
void BoxingKernel<T>::InferCopyRulesFromEqualAxis(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob,
    const int32_t axis, const std::vector<std::string>& src_bns,
    const std::vector<std::string>& dst_bns) const {
  // P.S This routine will be called only once, thus some performance
  // loss seems ok.
  std::map<const std::string*, int64_t> src_bn2concat_dim;
  std::map<const std::string*, int64_t> dst_bn2concat_dim;
  for (const std::string& bn : src_bns) {
    if (BnInOp2Blob(bn) == nullptr) { break; }
    CHECK(src_bn2concat_dim.emplace(&bn, (BnInOp2Blob(bn)->shape().At(axis)))
              .second);
  }
  for (const std::string& bn : dst_bns) {
    if (BnInOp2Blob(bn) == nullptr) { break; }
    CHECK(dst_bn2concat_dim.emplace(&bn, (BnInOp2Blob(bn)->shape().At(axis)))
              .second);
  }

  Blob* src_fst_blob = BnInOp2Blob(src_bns.front());
  const int64_t concat_dim_sz = (src_fst_blob->shape().NumAxes() > axis + 1)
                                    ? src_fst_blob->shape().Count(axis + 1)
                                    : 1;
  int64_t seg_cnt = (axis == 0) ? 1 : (src_fst_blob->shape().Count(0, axis));

  InferCopyRulesFromConcatDim(ctx, src_bn2concat_dim, dst_bn2concat_dim,
                              seg_cnt, concat_dim_sz, axis);
}

template<typename T>
void BoxingKernel<T>::CopyFromSrc2Dst(
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
    CopyDataId(ctx, src_bns, dst_bns, src_blobs, dst_blobs, src_concat_axis,
               dst_split_axis);
  }

  if (src_concat_axis == dst_split_axis) {
    InferCopyRulesFromEqualAxis(ctx, BnInOp2Blob, src_concat_axis, src_bns,
                                dst_bns);
  } else {
    InferCopyRulesFromUnequalAxis(ctx, src_blobs, dst_blobs, src_concat_axis,
                                  dst_split_axis);
  }
}

template<typename T>
void BoxingKernel<T>::FwCloneData(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob,
    const std::vector<std::string>& obns) const {
  int64_t copy_size = BnInOp2Blob(obns.front())->shape().elem_cnt() * sizeof(T);
  for (size_t i = 1; i < obns.size(); ++i) {
    if (BnInOp2Blob(obns.at(i)) == nullptr) { break; }
    BoxingCopy(ctx, false, BnInOp2Blob(obns.front()), BnInOp2Blob(obns.at(i)),
               0, 0, copy_size, false);
  }
}

template<typename T>
void BoxingKernel<T>::Forward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  auto boxing_conf = op_conf();
  auto kernel_conf = kernel_conf();
  if (boxing_conf.in_box_case() == BoxingOpConf::kConcatBox) {
    // concat-box copy rules: copy directly from input to output
    int32_t concat_axis = boxing_conf.concat_box().concat_axis();
    if (boxing_conf.out_box_case() == BoxingOpConf::kSplitBox) {
      int32_t split_axis = boxing_conf.split_box().split_axis();
      CopyFromSrc2Dst(BnInOp2Blob, kernel_conf.input_bns(),
                      kernel_conf.output_bns(), concat_axis, split_axis);
    } else if (boxing_conf.out_box_case() == BoxingOpConf::kCloneBox) {
      CopyFromSrc2Dst(BnInOp2Blob, kernel_conf.input_bns(), {"out_0"},
                      concat_axis, 0);
    } else {
      UNEXPECTED_RUN();
    }
  } else if (boxing_conf.in_box_case() == BoxingOpConf::kAddBox) {
    if (boxing_conf.out_box_case() == BoxingOpConf::kSplitBox) {
      GetSumFromSrcBlobsToDstBlob(ctx, BnInOp2Blob, kernel_conf.input_bns(),
                                  {"middle"});
      int32_t split_axis = boxing_conf.split_box().split_axis();
      CopyFromSrc2Dst(BnInOp2Blob, {"middle"}, kernel_conf.output_bns(), 0,
                      split_axis);
    } else if (boxing_conf.out_box_case() == BoxingOpConf::kCloneBox) {
      GetSumFromSrcBlobsToDstBlob(ctx, BnInOp2Blob, kernel_conf.input_bns(),
                                  {kernel_conf.output_bns().front()});
    } else {
      UNEXPECTED_RUN();
    }
  }
  if (boxing_conf.out_box_case() == BoxingOpConf::kCloneBox) {
    FwCloneData(ctx, BnInOp2Blob, kernel_conf.output_bns());
  }
}

}  // namespace oneflow
