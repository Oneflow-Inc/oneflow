#include "oneflow/core/kernel/boxing_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/operator/op_conf.pb.h"

namespace oneflow {

namespace {

PbRpf<std::string> ConstructPbRpf(const std::string& s) {
  PbRpf<std::string> ret;
  ret.Reserve(1);
  ret.Add()->assign(s);
  return ret;
}

template<typename T>
void CalcSumOfBlobs(DeviceCtx* ctx,
                    std::function<Blob*(const std::string&)> BnInOp2Blob,
                    const PbRpf<std::string>& src_bns,
                    const std::string& dst_bn) {
  const Blob* src_blob_0 = BnInOp2Blob(src_bns[0]);
  Blob* dst_blob = BnInOp2Blob(dst_bn);
  Memcpy<DeviceType::kCPU>(ctx, dst_blob->mut_dptr(), src_blob_0->dptr(),
                           src_blob_0->ByteSizeOfDataContentField());
  FOR_RANGE(size_t, i, 1, src_bns.size()) {
    Blob* src_blob_i = BnInOp2Blob(src_bns[i]);
    KernelUtil<DeviceType::kCPU, T>::Axpy(ctx, dst_blob->shape().elem_cnt(),
                                          1.0, src_blob_i->dptr<T>(), 1,
                                          dst_blob->mut_dptr<T>(), 1);
  }
}

void CopyFromFirstToOtherBlobs(
    DeviceCtx* ctx, std::function<Blob*(const std::string&)> BnInOp2Blob,
    const PbRpf<std::string>& bns,
    void (Blob::*Copy)(DeviceCtx*, const Blob*)) {
  const Blob* blob_0 = BnInOp2Blob(bns[0]);
  FOR_RANGE(size_t, i, 1, bns.size()) {
    (BnInOp2Blob(bns[i])->*Copy)(ctx, blob_0);
  }
}

void CopyDataContentFromFirstToOtherBlobs(
    DeviceCtx* ctx, std::function<Blob*(const std::string&)> BnInOp2Blob,
    const PbRpf<std::string>& bns) {
  CopyFromFirstToOtherBlobs(ctx, BnInOp2Blob, bns,
                            &Blob::CopyDataContentFrom<DeviceType::kCPU>);
}

void CopyDataIdFromFirstToOtherBlobs(
    DeviceCtx* ctx, std::function<Blob*(const std::string&)> BnInOp2Blob,
    const PbRpf<std::string>& bns) {
  CopyFromFirstToOtherBlobs(ctx, BnInOp2Blob, bns,
                            &Blob::CopyDataIdFrom<DeviceType::kCPU>);
}

template<typename Iter>
void CopyFromIterToIter(DeviceCtx* ctx, Iter& src_it, Iter& dst_it) {
  const char* src_ptr = nullptr;
  size_t src_size = 0;
  char* dst_ptr = nullptr;
  size_t dst_size = 0;
  while (true) {
    if (src_size == 0) { std::tie(src_ptr, src_size) = src_it.GetNext(); }
    if (dst_size == 0) { std::tie(dst_ptr, dst_size) = dst_it.GetNext(); }
    if (src_size == 0) {
      CHECK_EQ(src_size, dst_size);
      break;
    }
    size_t cp_size = std::min(src_size, dst_size);
    Memcpy<DeviceType::kCPU>(ctx, dst_ptr, src_ptr, cp_size);
    src_ptr += cp_size;
    src_size -= cp_size;
    dst_ptr += cp_size;
    dst_size -= cp_size;
  }
}

class DataContentIterator final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DataContentIterator);
  DataContentIterator() = delete;
  ~DataContentIterator() = default;

  DataContentIterator(std::function<Blob*(const std::string&)> BnInOp2Blob,
                      const PbRpf<std::string>* bns, int32_t axis) {
    BnInOp2Blob_ = BnInOp2Blob;
    seg_num_ = BnInOp2Blob(bns->Get(0))->shape().Count(0, axis);
    seg_idx_ = 0;
    bns_ = bns;
    bn_idx_ = 0;
    axis_ = axis;
  }

  std::tuple<char*, size_t> GetNext() {
    std::tuple<char*, size_t> ret(nullptr, 0);
    if (seg_idx_ == seg_num_) { return ret; }
    Blob* blob = BnInOp2Blob_(bns_->Get(bn_idx_));
    int64_t elem_num = blob->shape().Count(axis_);
    std::get<1>(ret) = elem_num * GetSizeOfDataType(blob->data_type());
    std::get<0>(ret) = blob->mut_dptr<char>() + seg_idx_ * std::get<1>(ret);
    bn_idx_ += 1;
    if (bn_idx_ == bns_->size()) {
      bn_idx_ = 0;
      seg_idx_ += 1;
    }
    return ret;
  }

 private:
  std::function<Blob*(const std::string&)> BnInOp2Blob_;
  int64_t seg_num_;
  int64_t seg_idx_;
  const PbRpf<std::string>* bns_;
  int32_t bn_idx_;
  int32_t axis_;
};

void ConcatSplitDataContent(
    DeviceCtx* ctx, std::function<Blob*(const std::string&)> BnInOp2Blob,
    const PbRpf<std::string>& concat_bns, int32_t concat_axis,
    const PbRpf<std::string>& split_bns, int32_t split_axis) {
  DataContentIterator concat_it(BnInOp2Blob, &concat_bns, concat_axis);
  DataContentIterator split_it(BnInOp2Blob, &split_bns, split_axis);
  CopyFromIterToIter(ctx, concat_it, split_it);
}

class DataIdIterator final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DataIdIterator);
  DataIdIterator() = delete;
  ~DataIdIterator() = default;

  DataIdIterator(std::function<Blob*(const std::string&)> BnInOp2Blob,
                 const PbRpf<std::string>* bns, int32_t axis) {
    BnInOp2Blob_ = BnInOp2Blob;
    bns_ = bns;
    bn_idx_ = 0;
    if (axis == 0) {
      bn_num_ = bns_->size();
    } else {
      bn_num_ = 1;
    }
  }

  std::tuple<char*, size_t> GetNext() {
    std::tuple<char*, size_t> ret(nullptr, 0);
    if (bn_idx_ == bn_num_) { return ret; }
    Blob* blob = BnInOp2Blob_(bns_->Get(bn_idx_++));
    std::get<0>(ret) = blob->mut_data_id();
    std::get<1>(ret) = blob->ByteSizeOfDataIdField();
    return ret;
  }

 private:
  std::function<Blob*(const std::string&)> BnInOp2Blob_;
  const PbRpf<std::string>* bns_;
  int32_t bn_idx_;
  int32_t bn_num_;
};

void ConcatSplitDataId(DeviceCtx* ctx,
                       std::function<Blob*(const std::string&)> BnInOp2Blob,
                       const PbRpf<std::string>& concat_bns,
                       int32_t concat_axis, const PbRpf<std::string>& split_bns,
                       int32_t split_axis) {
  DataIdIterator concat_it(BnInOp2Blob, &concat_bns, concat_axis);
  DataIdIterator split_it(BnInOp2Blob, &split_bns, split_axis);
  CopyFromIterToIter(ctx, concat_it, split_it);
  if (split_axis != 0) {
    CopyDataIdFromFirstToOtherBlobs(ctx, BnInOp2Blob, split_bns);
  }
}

}  // namespace

template<typename T>
void BoxingKernel<T>::VirtualKernelInit(const ParallelContext*) {
  const std::string& ibn_0 = kernel_conf().input_bns(0);
  const std::string& obn_0 = kernel_conf().output_bns(0);
  ibn_0_ = ConstructPbRpf(ibn_0);
  obn_0_ = ConstructPbRpf(obn_0);
}

template<typename T>
void BoxingKernel<T>::ForwardDataContent(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const BoxingOpConf& boxing_conf = op_conf().boxing_conf();
  if (boxing_conf.in_box_case() == BoxingOpConf::kConcatBox) {
    if (boxing_conf.out_box_case() == BoxingOpConf::kSplitBox) {
      ConcatSplitDataContent(
          ctx.device_ctx, BnInOp2Blob, kernel_conf().input_bns(),
          boxing_conf.concat_box().axis(), kernel_conf().output_bns(),
          boxing_conf.split_box().axis());
    } else if (boxing_conf.out_box_case() == BoxingOpConf::kCloneBox) {
      ConcatSplitDataContent(ctx.device_ctx, BnInOp2Blob,
                             kernel_conf().input_bns(),
                             boxing_conf.concat_box().axis(), obn_0_, 0);
      CopyDataContentFromFirstToOtherBlobs(ctx.device_ctx, BnInOp2Blob,
                                           kernel_conf().output_bns());
    } else {
      UNEXPECTED_RUN();
    }
  } else if (boxing_conf.in_box_case() == BoxingOpConf::kAddBox) {
    if (boxing_conf.out_box_case() == BoxingOpConf::kSplitBox) {
      CalcSumOfBlobs<T>(ctx.device_ctx, BnInOp2Blob, kernel_conf().input_bns(),
                        "middle");
      ConcatSplitDataContent(
          ctx.device_ctx, BnInOp2Blob, ConstructPbRpf("middle"), 0,
          kernel_conf().output_bns(), boxing_conf.split_box().axis());
    } else if (boxing_conf.out_box_case() == BoxingOpConf::kCloneBox) {
      CalcSumOfBlobs<T>(ctx.device_ctx, BnInOp2Blob, kernel_conf().input_bns(),
                        obn_0_.Get(0));
      CopyDataContentFromFirstToOtherBlobs(ctx.device_ctx, BnInOp2Blob,
                                           kernel_conf().output_bns());
    } else {
      UNEXPECTED_RUN();
    }
  } else {
    UNEXPECTED_RUN();
  }
}

template<typename T>
void BoxingKernel<T>::ForwardDataId(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const BoxingOpConf& boxing_conf = op_conf().boxing_conf();
  if (boxing_conf.in_box_case() == BoxingOpConf::kConcatBox) {
    if (boxing_conf.out_box_case() == BoxingOpConf::kSplitBox) {
      ConcatSplitDataId(ctx.device_ctx, BnInOp2Blob, kernel_conf().input_bns(),
                        boxing_conf.concat_box().axis(),
                        kernel_conf().output_bns(),
                        boxing_conf.split_box().axis());
    } else if (boxing_conf.out_box_case() == BoxingOpConf::kCloneBox) {
      ConcatSplitDataId(ctx.device_ctx, BnInOp2Blob, kernel_conf().input_bns(),
                        boxing_conf.concat_box().axis(), obn_0_, 0);
      CopyDataIdFromFirstToOtherBlobs(ctx.device_ctx, BnInOp2Blob,
                                      kernel_conf().output_bns());
    } else {
      UNEXPECTED_RUN();
    }
  } else if (boxing_conf.in_box_case() == BoxingOpConf::kAddBox) {
    if (boxing_conf.out_box_case() == BoxingOpConf::kSplitBox) {
      ConcatSplitDataId(ctx.device_ctx, BnInOp2Blob, ibn_0_, 0,
                        kernel_conf().output_bns(),
                        boxing_conf.split_box().axis());
    } else if (boxing_conf.out_box_case() == BoxingOpConf::kCloneBox) {
      CopyDataIdToAllOb(ctx.device_ctx, BnInOp2Blob,
                        BnInOp2Blob(ibn_0_.Get(0)));
    } else {
      UNEXPECTED_RUN();
    }
  } else {
    UNEXPECTED_RUN();
  }
}

ADD_CPU_DEFAULT_KERNEL_CREATOR(OperatorConf::kBoxingConf, BoxingKernel,
                               ARITHMETIC_DATA_TYPE_SEQ);

}  // namespace oneflow
