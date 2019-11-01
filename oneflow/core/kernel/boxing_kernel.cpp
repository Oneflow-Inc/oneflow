#include "oneflow/core/kernel/boxing_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/operator/op_conf_util.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/thread/thread_manager.h"
#include "oneflow/core/common/blocking_counter.h"

namespace oneflow {

namespace {

PbRpf<std::string> ConstructPbRpf(const std::string& s) {
  PbRpf<std::string> ret;
  ret.Reserve(1);
  ret.Add()->assign(s);
  return ret;
}

template<typename T>
void CalcSumOfBlobs(DeviceCtx* ctx, std::function<Blob*(const std::string&)> BnInOp2Blob,
                    const PbRpf<std::string>& src_bns, const std::string& dst_bn) {
  const Blob* src_blob_0 = BnInOp2Blob(src_bns.Get(0));
  Blob* dst_blob = BnInOp2Blob(dst_bn);
  Memcpy<DeviceType::kCPU>(ctx, dst_blob->mut_dptr(), src_blob_0->dptr(),
                           src_blob_0->ByteSizeOfDataContentField());
  FOR_RANGE(size_t, i, 1, src_bns.size()) {
    Blob* src_blob_i = BnInOp2Blob(src_bns.Get(i));
    KernelUtil<DeviceType::kCPU, T>::Axpy(ctx, dst_blob->static_shape().elem_cnt(), 1.0,
                                          src_blob_i->dptr<T>(), 1, dst_blob->mut_dptr<T>(), 1);
  }
}

void CopyFromFirstToOtherBlobs(DeviceCtx* ctx, std::function<Blob*(const std::string&)> BnInOp2Blob,
                               const PbRpf<std::string>& bns, CopyBlobFieldMthd Copy) {
  const Blob* blob_0 = BnInOp2Blob(bns.Get(0));
  FOR_RANGE(size_t, i, 1, bns.size()) { (BnInOp2Blob(bns.Get(i))->*Copy)(ctx, blob_0); }
}

class DataContentDesc final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DataContentDesc);
  DataContentDesc() = delete;
  ~DataContentDesc() = default;

  DataContentDesc(std::function<Blob*(const std::string&)> BnInOp2Blob,
                  const PbRpf<std::string>* bns, int32_t axis) {
    BnInOp2Blob_ = BnInOp2Blob;
    seg_num_ = BnInOp2Blob(bns->Get(0))->static_shape().Count(0, axis);
    elem_sum_.assign(bns->size(), 0);
    FOR_RANGE(size_t, i, 0, elem_sum_.size()) {
      elem_sum_[i] = BnInOp2Blob(bns->Get(i))->static_shape().Count(axis);
      if (i > 0) { elem_sum_[i] += elem_sum_[i - 1]; }
    }
    bns_ = bns;
    axis_ = axis;
  }

  size_t OneElemSize() const { return GetSizeOfDataType(BnInOp2Blob_(bns_->Get(0))->data_type()); }

  int64_t TotalElemNum() const { return seg_num_ * elem_sum_.back(); }

  std::tuple<int64_t, char*> CalcContinuousElemNumStartFrom(int64_t idx) const {
    std::tuple<int64_t, char*> ret(0, nullptr);
    int64_t seg_idx = idx / elem_sum_.back();
    int64_t idx_in_seg = idx % elem_sum_.back();
    auto elem_sum_it = std::upper_bound(elem_sum_.begin(), elem_sum_.end(), idx_in_seg);
    CHECK(elem_sum_it != elem_sum_.end());
    std::get<0>(ret) = *elem_sum_it - idx_in_seg;
    int64_t bn_idx = elem_sum_it - elem_sum_.begin();
    int64_t idx_in_blob = idx_in_seg;
    if (bn_idx > 0) { idx_in_blob -= elem_sum_[bn_idx - 1]; }
    Blob* blob = BnInOp2Blob_(bns_->Get(bn_idx));
    std::get<1>(ret) = blob->mut_dptr<char>()
                       + (seg_idx * blob->static_shape().Count(axis_) + idx_in_blob)
                             * GetSizeOfDataType(blob->data_type());
    return ret;
  }

 private:
  std::function<Blob*(const std::string&)> BnInOp2Blob_;
  int64_t seg_num_;
  std::vector<int64_t> elem_sum_;
  const PbRpf<std::string>* bns_;
  int32_t axis_;
};

void ConcatSplitPartDataContent(DeviceCtx* ctx, const DataContentDesc& in_desc,
                                const DataContentDesc& out_desc, int32_t part_id,
                                int32_t part_num) {
  size_t one_elem_size = in_desc.OneElemSize();
  BalancedSplitter bs(in_desc.TotalElemNum(), part_num);
  Range range = bs.At(part_id);
  int64_t in_idx = range.begin();
  int64_t in_elem_num = 0;
  char* in_ptr = nullptr;
  int64_t out_idx = range.begin();
  int64_t out_elem_num = 0;
  char* out_ptr = nullptr;
  while (in_elem_num > 0 || out_elem_num > 0 || in_idx < range.end() || out_idx < range.end()) {
    if (in_elem_num == 0) {
      std::tie(in_elem_num, in_ptr) = in_desc.CalcContinuousElemNumStartFrom(in_idx);
      in_elem_num = std::min(in_elem_num, range.end() - in_idx);
      if (in_elem_num == 0) { break; }
      in_idx += in_elem_num;
    }
    if (out_elem_num == 0) {
      std::tie(out_elem_num, out_ptr) = out_desc.CalcContinuousElemNumStartFrom(out_idx);
      out_elem_num = std::min(out_elem_num, range.end() - out_idx);
      if (out_elem_num == 0) { break; }
      out_idx += out_elem_num;
    }
    int64_t copy_elem_num = std::min(in_elem_num, out_elem_num);
    size_t copy_size = copy_elem_num * one_elem_size;
    Memcpy<DeviceType::kCPU>(ctx, out_ptr, in_ptr, copy_size);
    in_elem_num -= copy_elem_num;
    out_elem_num -= copy_elem_num;
    in_ptr += copy_size;
    out_ptr += copy_size;
  }
  CHECK_EQ(in_elem_num, 0);
  CHECK_EQ(out_elem_num, 0);
  CHECK_EQ(in_idx, range.end());
  CHECK_EQ(out_idx, range.end());
}

void ConcatSplitDataContent(DeviceCtx* ctx, std::function<Blob*(const std::string&)> BnInOp2Blob,
                            const PbRpf<std::string>& concat_bns, int32_t concat_axis,
                            const PbRpf<std::string>& split_bns, int32_t split_axis) {
  DataContentDesc in_desc(BnInOp2Blob, &concat_bns, concat_axis);
  DataContentDesc out_desc(BnInOp2Blob, &split_bns, split_axis);
  CHECK_EQ(in_desc.TotalElemNum(), out_desc.TotalElemNum());
  CHECK_EQ(in_desc.OneElemSize(), out_desc.OneElemSize());
  static const size_t min_byte_one_part = 128;
  int32_t part_num = in_desc.TotalElemNum() * in_desc.OneElemSize() / min_byte_one_part;
  part_num = std::min(part_num, Global<ThreadMgr>::Get()->compute_thread_pool()->thread_num());
  if (part_num >= 2) {
    BlockingCounter bc(part_num);
    FOR_RANGE(int32_t, part_id, 0, part_num) {
      Global<ThreadMgr>::Get()->compute_thread_pool()->AddWork(
          [&ctx, &in_desc, &out_desc, part_id, &part_num, &bc]() {
            ConcatSplitPartDataContent(ctx, in_desc, out_desc, part_id, part_num);
            bc.Decrease();
          });
    }
    bc.WaitUntilCntEqualZero();
  } else {
    ConcatSplitPartDataContent(ctx, in_desc, out_desc, 0, 1);
  }
}

template<typename Iter>
void ConcatSplitField(DeviceCtx* ctx, std::function<Blob*(const std::string&)> BnInOp2Blob,
                      const PbRpf<std::string>& concat_bns, int32_t concat_axis,
                      const PbRpf<std::string>& split_bns, int32_t split_axis) {
  Iter concat_it(BnInOp2Blob, &concat_bns, concat_axis);
  Iter split_it(BnInOp2Blob, &split_bns, split_axis);
  CopyFromIterToIter<DeviceType::kCPU>(ctx, concat_it, split_it);
  if (split_axis != 0) {
    CopyFromFirstToOtherBlobs(ctx, BnInOp2Blob, split_bns, Iter::GetCopyBlobFieldMthd());
  }
}

int32_t MaxColIdInBlobs(std::function<Blob*(const std::string&)> BnInOp2Blob,
                        const PbRpf<std::string>& bns) {
  int32_t max_col_id_in_bns = 0;
  for (const std::string& bn : bns) {
    Blob* blob = BnInOp2Blob(bn);
    max_col_id_in_bns = std::max(max_col_id_in_bns, blob->col_id());
  }
  return max_col_id_in_bns;
}

void SetBlobsColId(std::function<Blob*(const std::string&)> BnInOp2Blob,
                   const PbRpf<std::string>& bns, int32_t col_id) {
  for (const std::string& bn : bns) { BnInOp2Blob(bn)->set_col_id(col_id); }
}

void ConcatSplitColId(std::function<Blob*(const std::string&)> BnInOp2Blob,
                      const PbRpf<std::string>& input_bns, const PbRpf<std::string>& output_bns) {
  auto in_iter = input_bns.begin();
  auto out_iter = output_bns.begin();
  int64_t in_data_num = BnInOp2Blob(*in_iter)->static_shape().At(0);
  int64_t out_data_num = BnInOp2Blob(*out_iter)->static_shape().At(0);
  int32_t max_col_id = BnInOp2Blob(*in_iter)->col_id();
  while (in_iter != input_bns.end() && out_iter != input_bns.end()) {
    if (in_data_num < out_data_num) {
      ++in_iter;
      in_data_num += BnInOp2Blob(*in_iter)->static_shape().At(0);
      max_col_id = std::max(max_col_id, BnInOp2Blob(*in_iter)->col_id());
    } else if (in_data_num > out_data_num) {
      BnInOp2Blob(*out_iter)->set_col_id(max_col_id);
      max_col_id = BnInOp2Blob(*in_iter)->col_id();
      ++out_iter;
      out_data_num += BnInOp2Blob(*out_iter)->static_shape().At(0);
    } else {
      BnInOp2Blob(*out_iter)->set_col_id(max_col_id);
      ++in_iter;
      in_data_num += BnInOp2Blob(*in_iter)->static_shape().At(0);
      max_col_id = BnInOp2Blob(*in_iter)->col_id();
      ++out_iter;
      out_data_num += BnInOp2Blob(*out_iter)->static_shape().At(0);
    }
  }
}

}  // namespace

template<typename T>
void BoxingKernel<T>::VirtualKernelInit() {
  const std::string& ibn_0 = op_attribute().input_bns(0);
  const std::string& obn_0 = op_attribute().output_bns(0);
  ibn_0_ = ConstructPbRpf(ibn_0);
  obn_0_ = ConstructPbRpf(obn_0);
  CHECK_EQ(kernel_conf().need_do_opaque_header(), false);
}

template<typename T>
void BoxingKernel<T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const BoxingOpConf& boxing_conf = op_conf().boxing_conf();
  if (boxing_conf.in_box_case() == BoxingOpConf::kConcatBox) {
    if (boxing_conf.out_box_case() == BoxingOpConf::kSplitBox) {
      ConcatSplitDataContent(ctx.device_ctx, BnInOp2Blob, op_attribute().input_bns(),
                             boxing_conf.concat_box().axis(), op_attribute().output_bns(),
                             boxing_conf.split_box().axis());
    } else if (boxing_conf.out_box_case() == BoxingOpConf::kCloneBox) {
      ConcatSplitDataContent(ctx.device_ctx, BnInOp2Blob, op_attribute().input_bns(),
                             boxing_conf.concat_box().axis(), obn_0_, 0);
      CopyFromFirstToOtherBlobs(ctx.device_ctx, BnInOp2Blob, op_attribute().output_bns(),
                                DataContentIterator::GetCopyBlobFieldMthd());
    } else {
      UNIMPLEMENTED();
    }
  } else if (boxing_conf.in_box_case() == BoxingOpConf::kAddBox) {
    if (boxing_conf.out_box_case() == BoxingOpConf::kSplitBox) {
      CalcSumOfBlobs<T>(ctx.device_ctx, BnInOp2Blob, op_attribute().input_bns(), "middle");
      ConcatSplitDataContent(ctx.device_ctx, BnInOp2Blob, ConstructPbRpf("middle"), 0,
                             op_attribute().output_bns(), boxing_conf.split_box().axis());
    } else if (boxing_conf.out_box_case() == BoxingOpConf::kCloneBox) {
      CalcSumOfBlobs<T>(ctx.device_ctx, BnInOp2Blob, op_attribute().input_bns(), obn_0_.Get(0));
      CopyFromFirstToOtherBlobs(ctx.device_ctx, BnInOp2Blob, op_attribute().output_bns(),
                                DataContentIterator::GetCopyBlobFieldMthd());
    } else {
      UNIMPLEMENTED();
    }
  } else {
    UNIMPLEMENTED();
  }
}

template<typename T>
template<typename Iter>
void BoxingKernel<T>::ForwardField(const KernelCtx& ctx,
                                   std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const BoxingOpConf& boxing_conf = op_conf().boxing_conf();
  if (boxing_conf.in_box_case() == BoxingOpConf::kConcatBox) {
    if (boxing_conf.out_box_case() == BoxingOpConf::kSplitBox) {
      ConcatSplitField<Iter>(ctx.device_ctx, BnInOp2Blob, op_attribute().input_bns(),
                             boxing_conf.concat_box().axis(), op_attribute().output_bns(),
                             boxing_conf.split_box().axis());
    } else if (boxing_conf.out_box_case() == BoxingOpConf::kCloneBox) {
      ConcatSplitField<Iter>(ctx.device_ctx, BnInOp2Blob, op_attribute().input_bns(),
                             boxing_conf.concat_box().axis(), obn_0_, 0);
      CopyFromFirstToOtherBlobs(ctx.device_ctx, BnInOp2Blob, op_attribute().output_bns(),
                                Iter::GetCopyBlobFieldMthd());
    } else {
      UNIMPLEMENTED();
    }
  } else if (boxing_conf.in_box_case() == BoxingOpConf::kAddBox) {
    if (boxing_conf.out_box_case() == BoxingOpConf::kSplitBox) {
      ConcatSplitField<Iter>(ctx.device_ctx, BnInOp2Blob, ibn_0_, 0, op_attribute().output_bns(),
                             boxing_conf.split_box().axis());
    } else if (boxing_conf.out_box_case() == BoxingOpConf::kCloneBox) {
      CopyField(ctx.device_ctx, BnInOp2Blob, BnInOp2Blob(ibn_0_.Get(0)),
                op_attribute().output_bns(), Iter::GetCopyBlobFieldMthd());
    } else {
      UNIMPLEMENTED();
    }
  } else {
    UNIMPLEMENTED();
  }
}

template<typename T>
void BoxingKernel<T>::ForwardDataId(const KernelCtx& ctx,
                                    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  ForwardField<DataIdIterator>(ctx, BnInOp2Blob);
}

template<typename T>
void BoxingKernel<T>::ForwardColNum(const KernelCtx& ctx,
                                    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  ForwardField<ColNumIterator>(ctx, BnInOp2Blob);
  SetMaxColId(ctx, BnInOp2Blob);
  SetColId(ctx, BnInOp2Blob);
}

template<typename T>
void BoxingKernel<T>::ForwardDim0ValidNum(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  TODO();
}

template<typename T>
void BoxingKernel<T>::ForwardDim1ValidNum(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  ForwardField<Dim1ValidNumIterator>(ctx, BnInOp2Blob);
}

template<typename T>
void BoxingKernel<T>::ForwardDim2ValidNum(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  ForwardField<Dim2ValidNumIterator>(ctx, BnInOp2Blob);
}

template<typename T>
void BoxingKernel<T>::ForwardRecordIdInDevicePiece(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  // record indexes is not allowed to be passed through boxing op
  UNIMPLEMENTED();
}

template<typename T>
void BoxingKernel<T>::SetColId(const KernelCtx& ctx,
                               std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const BoxingOpConf& boxing_conf = op_conf().boxing_conf();
  if (boxing_conf.in_box_case() == BoxingOpConf::kConcatBox
      && boxing_conf.concat_box().axis() == 0) {
    if (boxing_conf.out_box_case() == BoxingOpConf::kSplitBox
        && boxing_conf.split_box().axis() == 0) {
      ConcatSplitColId(BnInOp2Blob, op_attribute().input_bns(), op_attribute().output_bns());
    } else {
      SetBlobsColId(BnInOp2Blob, op_attribute().output_bns(),
                    MaxColIdInBlobs(BnInOp2Blob, op_attribute().input_bns()));
    }
  } else {
    SetBlobsColId(BnInOp2Blob, op_attribute().output_bns(),
                  BnInOp2Blob(op_attribute().input_bns(0))->col_id());
  }
}

template<typename T>
void BoxingKernel<T>::SetMaxColId(const KernelCtx& ctx,
                                  std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  for (const std::string& obn : op_attribute().output_bns()) {
    int32_t max_col_num_in_blob = 0;
    Blob* out_blob = BnInOp2Blob(obn);
    FOR_RANGE(int32_t, i, 0, out_blob->static_shape().At(0)) {
      max_col_num_in_blob = std::max(max_col_num_in_blob, out_blob->col_num(i));
    }
    out_blob->set_max_col_id(max_col_num_in_blob - 1);
  }
}

ADD_CPU_DEFAULT_KERNEL_CREATOR(OperatorConf::kBoxingConf, BoxingKernel, ARITHMETIC_DATA_TYPE_SEQ);

}  // namespace oneflow
