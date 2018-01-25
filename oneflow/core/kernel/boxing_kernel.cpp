#include "oneflow/core/kernel/boxing_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/operator/op_conf.pb.h"

namespace oneflow {

namespace {

using BlobFieldCopyMthd = void (Blob::*)(DeviceCtx*, const Blob*);

PbRpf<std::string> ConstructPbRpf(const std::string& s) {
  PbRpf<std::string> ret;
  ret.Reserve(1);
  ret.Add()->assign(s);
  return ret;
}

bool IsValidBlobInBlobs(const Blob* blob, ) {

}

template<typename T>
void CalcSumOfBlobs(DeviceCtx* ctx,
                    std::function<Blob*(const std::string&)> BnInOp2Blob,
                    const PbRpf<std::string>& src_bns,
                    const std::string& dst_bn) {
  const Blob* src_blob_0 = BnInOp2Blob(src_bns.Get(0));
  Blob* dst_blob = BnInOp2Blob(dst_bn);
  Memcpy<DeviceType::kCPU>(ctx, dst_blob->mut_dptr(), src_blob_0->dptr(),
                           src_blob_0->ByteSizeOfDataContentField());
  FOR_RANGE(size_t, i, 1, src_bns.size()) {
    Blob* src_blob_i = BnInOp2Blob(src_bns.Get(i));
    KernelUtil<DeviceType::kCPU, T>::Axpy(ctx, dst_blob->shape().elem_cnt(),
                                          1.0, src_blob_i->dptr<T>(), 1,
                                          dst_blob->mut_dptr<T>(), 1);
  }
}

void CopyFromFirstToOtherBlobs(
    DeviceCtx* ctx, std::function<Blob*(const std::string&)> BnInOp2Blob,
    const PbRpf<std::string>& bns,
    BlobFieldCopyMthd Copy) {
  const Blob* blob_0 = BnInOp2Blob(bns.Get(0));
  FOR_RANGE(size_t, i, 1, bns.size()) {
    (BnInOp2Blob(bns.Get(i))->*Copy)(ctx, blob_0);
  }
}

template<typename Iter>
void CopyFromIterToIter(DeviceCtx* ctx, Iter& src_it, Iter& dst_it) {
  while(src_it.HasNext()) {
    dst_it.SetNext(src_it.GetNext());
  }
  CHECK(!dst_it.HasNext());
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

  bool HasNext() {
    return seg_idx_ != seg_num_;
  }

  void SetNext(DeviceCtx* ctx, std::tuple<char*, size_t> next_data){

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

  static BlobFieldCopyMthd GetCopyFunc() const {
    return &Blob::CopyDataContentFrom<DeviceType::kCPU>;
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
  CopyFromIterToIter<DataContentIterator>(ctx, concat_it, split_it);
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

  bool HasNext() {

  }

  void SetNext(DeviceCtx* ctx, std::tuple<char*, size_t> next_data){

  }

  std::tuple<char*, size_t> GetNext() {
    std::tuple<char*, size_t> ret(nullptr, 0);
    if (bn_idx_ == bn_num_) { return ret; }
    Blob* blob = BnInOp2Blob_(bns_->Get(bn_idx_++));
    std::get<0>(ret) = blob->mut_data_id();
    std::get<1>(ret) = blob->ByteSizeOfDataIdField();
    return ret;
  }

  static BlobFieldCopyMthd GetCopyFunc() const {
    return &Blob::CopyDataIdFrom<DeviceType::kCPU>;
  }

 private:
  std::function<Blob*(const std::string&)> BnInOp2Blob_;
  const PbRpf<std::string>* bns_;
  int32_t bn_idx_;
  int32_t bn_num_;
};

class ColNumIterator final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ColNumIterator);
  ColNumIterator() = delete;
  ~ColNumIterator() = default;

  ColNumIterator(std::function<Blob*(const std::string&)> BnInOp2Blob,
                 const PbRpf<std::string>* bns, int32_t axis) {
    BnInOp2Blob_ = BnInOp2Blob;
    bns_ = bns;
    bn_idx_ = 0;
    if (axis == 0) {
      bn_num_ = bns_->size();
    } else {
      bn_num_ = 1;
    }
    col_num_idx_ = 0;
  }

  bool HasNext() {

  }

  void SetNext(int32_t next_col_num){

  }

  int32_t GetNext() {
    if (bn_idx_ == bn_num_) { return -1; }
    Blob* blob = BnInOp2Blob_(bns_->Get(bn_idx_++));
    std::get<0>(ret) = blob->mut_data_id();
    std::get<1>(ret) = blob->ByteSizeOfDataIdField();
    return ret;
  }

  static BlobFieldCopyMthd GetCopyFunc() const {
    return &Blob::CopyColNumFrom<DeviceType::kCPU>;
  }

 private:
  std::function<Blob*(const std::string&)> BnInOp2Blob_;
  const PbRpf<std::string>* bns_;
  int32_t bn_idx_;
  int32_t bn_num_;
  int32_t col_num_idx_;
};

template<typename Iter>
void ConcatSplitField(DeviceCtx* ctx,
                      std::function<Blob*(const std::string&)> BnInOp2Blob,
                      const PbRpf<std::string>& concat_bns,
                      int32_t concat_axis, const PbRpf<std::string>& split_bns,
                      int32_t split_axis) {
  Iter concat_it(BnInOp2Blob, &concat_bns, concat_axis);
  Iter split_it(BnInOp2Blob, &split_bns, split_axis);
  CopyFromIterToIter<Iter>(ctx, concat_it, split_it);
  if (split_axis != 0) {
    CopyFromFirstToOtherBlobs(ctx, BnInOp2Blob, split_bns, Iter::GetCopyFunc());
  }
}

int32_t MaxColIdInBlobs(std::function<Blob*(const std::string&)> BnInOp2Blob,
                        const PbRpf<std::string>& bns) {
  int32_t max_col_id_in_bns = 0;
  for(const std::string& bn : bns) {
    Blob* blob = BnInOp2Blob(bn);
    max_col_id_in_bns = std::max(max_col_id_in_bns, blob->col_id());
  }
}

void SetBlobsColId(std::function<Blob*(const std::string&)> BnInOp2Blob,
                   const PbRpf<std::string>& bns, int32_t col_id) {
  for(const std::string& bn : bns) { BnInOp2Blob(bn)->set_col_id(col_id); }
}

void ConcatSplitColId(std::function<Blob*(const std::string&)> BnInOp2Blob,
                      const PbRpf<std::string>& input_bns, 
                      const PbRpf<std::string>& output_bns) {
  auto in_iter = input_bns.begin();
  auto out_iter = input_bns.begin();
  int64_t in_size = BnInOp2Blob(*in_iter)->shape().At(0);
  int64_t out_size = BnInOp2Blob(*out_iter)->shape().At(0);
  int32_t max_col_id = BnInOp2Blob(*in_iter)->col_id();
  while(in_iter!=input_bns.end() && out_iter!=input_bns.end()){
    if(in_size < out_size) {
      ++in_iter;
      in_size += BnInOp2Blob(*in_iter)->shape().At(0);
      max_col_id = std::max(max_col_id, BnInOp2Blob(*in_iter)->col_id());
    } else if (in_size > out_size) {
      BnInOp2Blob(*out_iter)->set_col_id(max_col_id);
      max_col_id = BnInOp2Blob(*in_iter)->col_id();
      ++out_iter;
      out_size += BnInOp2Blob(*out_iter)->shape().At(0);
    } else {
      BnInOp2Blob(*out_iter)->set_col_id(max_col_id);
      ++in_iter;
      in_size += BnInOp2Blob(*in_iter)->shape().At(0);
      max_col_id = BnInOp2Blob(*in_iter)->col_id();
      ++out_iter;
      out_size += BnInOp2Blob(*out_iter)->shape().At(0);
    }
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
      CopyFromFirstToOtherBlobs(ctx.device_ctx, BnInOp2Blob,
                                kernel_conf().output_bns(), 
                                &Blob::CopyDataContentFrom<DeviceType::kCPU>);
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
      CopyFromFirstToOtherBlobs(ctx.device_ctx, BnInOp2Blob,
                                kernel_conf().output_bns(), 
                                &Blob::CopyDataContentFrom<DeviceType::kCPU>);
    } else {
      UNEXPECTED_RUN();
    }
  } else {
    UNEXPECTED_RUN();
  }
}

template<typename T, typename Iter>
void BoxingKernel<T>::ForwardField(
    const KernelCtx& ctx, 
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const BoxingOpConf& boxing_conf = op_conf().boxing_conf();
  if (boxing_conf.in_box_case() == BoxingOpConf::kConcatBox) {
    if (boxing_conf.out_box_case() == BoxingOpConf::kSplitBox) {
      ConcatSplitField<Iter>(ctx.device_ctx, BnInOp2Blob, 
                        kernel_conf().input_bns(),
                        boxing_conf.concat_box().axis(),
                        kernel_conf().output_bns(),
                        boxing_conf.split_box().axis());
    } else if (boxing_conf.out_box_case() == BoxingOpConf::kCloneBox) {
      ConcatSplitField<Iter>(ctx.device_ctx, BnInOp2Blob, kernel_conf().input_bns(),
                        boxing_conf.concat_box().axis(), obn_0_, 0);
      CopyFromFirstToOtherBlobs(ctx.device_ctx, BnInOp2Blob,
                                kernel_conf().output_bns(), Iter::GetCopyFunc());
    } else {
      UNEXPECTED_RUN();
    }
  } else if (boxing_conf.in_box_case() == BoxingOpConf::kAddBox) {
    if (boxing_conf.out_box_case() == BoxingOpConf::kSplitBox) {
      ConcatSplitField<Iter>(ctx.device_ctx, BnInOp2Blob, ibn_0_, 0,
                        kernel_conf().output_bns(),
                        boxing_conf.split_box().axis());
    } else if (boxing_conf.out_box_case() == BoxingOpConf::kCloneBox) {
      CopyField(ctx.device_ctx, BnInOp2Blob, BnInOp2Blob(ibn_0_.Get(0)),
                kernel_conf().output_bns(), Iter::GetCopyFunc());
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
  ForwardField<T, DataIdIterator>(ctx, BnInOp2Blob);
}

template<typename T>
void BoxingKernel<T>::ForwardColNum(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  ForwardField<T, ColNumIterator>(ctx, BnInOp2Blob);
}

template<typename T>
void BoxingKernel<T>::SetColId(const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const BoxingOpConf& boxing_conf = op_conf().boxing_conf();
  if (boxing_conf.in_box_case() == BoxingOpConf::kConcatBox
      && boxing_conf.concat_box().axis() == 0){
    if (boxing_conf.out_box_case() == BoxingOpConf::kSplitBox
        && boxing_conf.split_box().axis() == 0) {
      ConcatSplitColId(BnInOp2Blob, kernel_conf().input_bns(), 
                       kernel_conf().output_bns());
    } else {
      SetBlobsColId(BnInOp2Blob, kernel_conf().output_bns(),
                    MaxColIdInBlobs(BnInOp2Blob, kernel_conf().input_bns()));
    }
  } else {
    SetBlobsColId(BnInOp2Blob, kernel_conf().output_bns(), 
                    BnInOp2Blob(kernel_conf().input_bns(0))->col_id());
  }
}
 
template<typename T>
void BoxingKernel<T>::SetMaxColId(const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* out_blob = nullptr;
  int32_t max_col_num_in_blob = 0;
  for(const std::string& obn : kernel_conf().output_bns()) {
    max_col_num_in_blob = 0;
    out_blob = BnInOp2Blob(obn);
    FOR_RANGE(int32_t, i, 0, out_blob->shape().at(0)) {
      max_col_num_in_blob = std::max(max_col_num_in_blob, out_blob->col_num(i));
    }
    out_blob->set_max_col_id(max_col_num_in_blob - 1);
  }
}

ADD_CPU_DEFAULT_KERNEL_CREATOR(OperatorConf::kBoxingConf, BoxingKernel,
                               ARITHMETIC_DATA_TYPE_SEQ);

}  // namespace oneflow
