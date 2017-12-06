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
  FOR_RANGE(size_t, i, 1, src_bns.size()) {
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
                                 const int64_t dst_offset, size_t copy_bytesize,
                                 bool need_swap) const {
  if (is_data_id) {
    Memcpy<DeviceType::kCPU>(ctx.device_ctx,
                             dst_blob->mut_data_id() + dst_offset,
                             src_blob->data_id() + src_offset, copy_bytesize,
                             cudaMemcpyKind::cudaMemcpyHostToHost);
  } else if (need_swap) {
    Memcpy<DeviceType::kCPU>(ctx.device_ctx,
                             src_blob->mut_dptr<char>() + src_offset,
                             dst_blob->dptr<char>() + dst_offset, copy_bytesize,
                             cudaMemcpyKind::cudaMemcpyHostToHost);
  } else {
    Memcpy<DeviceType::kCPU>(ctx.device_ctx,
                             dst_blob->mut_dptr<char>() + dst_offset,
                             src_blob->dptr<char>() + src_offset, copy_bytesize,
                             cudaMemcpyKind::cudaMemcpyHostToHost);
  }
}

template<typename T>
void BoxingKernel<T>::CopyDataId(const KernelCtx& ctx,
                                 std::vector<Blob*>& src_blobs,
                                 std::vector<Blob*>& dst_blobs,
                                 const int32_t src_concat_axis,
                                 const int32_t dst_split_axis) const {
  size_t data_id_bytesize = JobDesc::Singleton()->SizeOfOneDataId();
  if (src_concat_axis == 0 || dst_split_axis == 0) {
    int64_t src_idx = 0;
    int64_t dst_idx = 0;
    int64_t src_offset = 0;
    int64_t dst_offset = 0;
    while (src_idx < src_blobs.size() && dst_idx < dst_blobs.size()) {
      int64_t src_dim = src_blobs.at(src_idx)->shape().At(0);
      int64_t dst_dim = dst_blobs.at(dst_idx)->shape().At(0);
      int64_t copy_size = std::min(src_dim - src_offset, dst_dim - dst_offset);

      BoxingCopy(ctx, true, src_blobs.at(src_idx), dst_blobs.at(dst_idx),
                 src_offset * data_id_bytesize, dst_offset * data_id_bytesize,
                 static_cast<size_t>(copy_size * data_id_bytesize), false);

      src_offset += copy_size;
      if (src_offset == src_dim) {
        src_offset = 0;
        ++src_idx;
      }

      dst_offset += copy_size;
      if (dst_offset == dst_dim) {
        dst_offset = 0;
        ++dst_idx;
      }
    }
  }

  if (dst_split_axis > 0) {
    // add copy rules from first dst blob to all dst blobs
    FOR_RANGE(size_t, i, 1, dst_blobs.size()) {
      BoxingCopy(ctx, true, dst_blobs.front(), dst_blobs.at(i), 0, 0,
                 dst_blobs.front()->ByteSizeOfDataIdField(), false);
    }
  }
}

template<typename T>
void BoxingKernel<T>::DoUnequalAxisCopy(const KernelCtx& ctx,
                                        std::vector<Blob*>& src_blobs,
                                        std::vector<Blob*>& dst_blobs,
                                        const BoxingInfo& src_info,
                                        const BoxingInfo& dst_info,
                                        bool need_swap) const {
  std::vector<int64_t> dst_blob_offset(dst_blobs.size(), 0);
  FOR_RANGE(size_t, src_idx, 0, src_blobs.size()) {
    int64_t dst_segs_in_src_seg =
        src_info.size_of_subseg(src_idx) / dst_info.size_of_per_seg();
    int64_t src_seg_offset = 0;
    FOR_RANGE(size_t, dst_seg_idx, 0, dst_segs_in_src_seg) {
      FOR_RANGE(size_t, dst_idx, 0, dst_blobs.size()) {
        BoxingCopy(
            ctx, src_blobs.at(src_idx), dst_blobs.at(dst_idx),
            src_seg_offset * sizeof(T), dst_blob_offset[dst_idx] * sizeof(T),
            static_cast<size_t>(dst_info.size_of_subseg(dst_idx) * sizeof(T)),
            need_swap);
        src_seg_offset += dst_info.size_of_subseg(dst_idx);
        dst_blob_offset[dst_idx] += dst_info.size_of_subseg(dst_idx);
      }
    }
  }
}

template<typename T>
void BoxingKernel<T>::BoxingCopyForUnequalAxis(const KernelCtx& ctx,
                                               std::vector<Blob*>& src_blobs,
                                               std::vector<Blob*>& dst_blobs,
                                               const int32_t concat_axis,
                                               const int32_t split_axis) const {
  const BoxingKernelConf& kernel_conf = this->kernel_conf().boxing_conf();
  const BoxingInfo& in_info = kernel_conf.in_info();
  const BoxingInfo& out_info = kernel_conf.out_info();
  if (concat_axis > split_axis) {
    DoUnequalAxisCopy(ctx, dst_blobs, src_blobs, out_info, in_info, true);
  } else {
    DoUnequalAxisCopy(ctx, src_blobs, dst_blobs, in_info, out_info, false);
  }
}

template<typename T>
void BoxingKernel<T>::BoxingCopyForEqualAxis(const KernelCtx& ctx,
                                             std::vector<Blob*>& src_blobs,
                                             std::vector<Blob*>& dst_blobs,
                                             const int32_t axis) const {
  const BoxingKernelConf& kernel_conf = this->kernel_conf().boxing_conf();
  const BoxingInfo& in_info = kernel_conf.in_info();
  const BoxingInfo& out_info = kernel_conf.out_info();
  int64_t src_offset = 0;
  for (size_t src_idx = 0, dst_idx = 0;
       src_idx != src_blobs.size() && dst_idx != dst_blobs.size();) {
    int64_t dst_offset = 0;
    while (dst_offset < out_info.size_of_subseg(dst_idx)) {
      int64_t copy_size =
          std::min(in_info.size_of_subseg(src_idx) - src_offset,
                   out_info.size_of_subseg(dst_idx) - dst_offset);
      BoxingCopy(ctx, false, src_blobs.at(src_idx), dst_blobs.at(dst_idx),
                 src_offset * sizeof(T), dst_offset * sizeof(T),
                 static_cast<size_t>(copy_size * sizeof(T)), false);
      src_offset += copy_size;
      dst_offset += copy_size;
      if (src_offset == in_info.size_of_subseg(src_idx)) {
        if (++src_idx == src_blobs.size()) {
          CHECK_EQ(dst_offset, out_info.size_of_subseg(dst_idx));
          break;
        }
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
    src_blobs.emplace_back(BnInOp2Blob(bn));
  }
  for (const std::string& bn : dst_bns) {
    dst_blobs.emplace_back(BnInOp2Blob(bn));
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
    const std::vector<std::string>& obns, bool is_data_flow) const {
  Blob* first_out_blob = BnInOp2Blob(obns.front());
  int64_t copy_size = first_out_blob->shape().elem_cnt() * sizeof(T);
  bool need_copy_data_id = is_data_flow && first_out_blob->has_data_id();
  FOR_RANGE(size_t, i, 1, obns.size()) {
    BoxingCopy(ctx, false, first_out_blob, BnInOp2Blob(obns.at(i)), 0, 0,
               static_cast<size_t>(copy_size), false);
    if (need_copy_data_id) {
      BoxingCopy(ctx, true, first_out_blob, BnInOp2Blob(obns.at(i)), 0, 0,
                 first_out_blob->ByteSizeOfDataIdField(), false);
    }
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
      CopyFromFirstBlob2OtherBlobs(ctx, BnInOp2Blob, kernel_conf.output_bns(),
                                   true);
    } else {
      UNEXPECTED_RUN();
    }
  } else if (boxing_conf.in_box_case() == BoxingOpConf::kAddBox) {
    if (boxing_conf.out_box_case() == BoxingOpConf::kSplitBox) {
      GetSumFromSrcBlobsToDstBlob(ctx, BnInOp2Blob, kernel_conf.input_bns(),
                                  {"middle"});
      CopyFromSrcBlobs2DstBlobs(ctx, BnInOp2Blob, {"middle"},
                                kernel_conf.output_bns(), 0,
                                boxing_conf.split_box().axis());
    } else if (boxing_conf.in_box_case() == BoxingOpConf::kCloneBox) {
      GetSumFromSrcBlobsToDstBlob(ctx, BnInOp2Blob, kernel_conf.input_bns(),
                                  {kernel_conf.output_bns(0)});
      CopyFromFirstBlob2OtherBlobs(ctx, BnInOp2Blob, kernel_conf.output_bns(),
                                   false);
    } else {
      UNEXPECTED_RUN();
    }
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
                                       ARITHMETIC_DATA_TYPE_SEQ)};
  return creators.at(GetHashKey(JobDesc::Singleton()->DefaultDataType()))();
}

}  // namespace

}  // namespace oneflow
