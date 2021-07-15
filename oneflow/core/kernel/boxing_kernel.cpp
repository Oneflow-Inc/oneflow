/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
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
                           src_blob_0->ByteSizeOfBlobBody());
  FOR_RANGE(size_t, i, 1, src_bns.size()) {
    Blob* src_blob_i = BnInOp2Blob(src_bns.Get(i));
    KernelUtil<DeviceType::kCPU, T>::Axpy(ctx, dst_blob->static_shape().elem_cnt(), 1.0,
                                          src_blob_i->dptr<T>(), 1, dst_blob->mut_dptr<T>(), 1);
  }
}

template<>
void CalcSumOfBlobs<float16>(DeviceCtx* ctx, std::function<Blob*(const std::string&)> BnInOp2Blob,
                             const PbRpf<std::string>& src_bns, const std::string& dst_bn) {
  const Blob* src_blob_0 = BnInOp2Blob(src_bns.Get(0));
  Blob* dst_blob = BnInOp2Blob(dst_bn);
  Memcpy<DeviceType::kCPU>(ctx, dst_blob->mut_dptr(), src_blob_0->dptr(),
                           src_blob_0->ByteSizeOfBlobBody());
  FOR_RANGE(size_t, i, 1, src_bns.size()) {
    Blob* src_blob_i = BnInOp2Blob(src_bns.Get(i));
    FOR_RANGE(int, i, 0, dst_blob->static_shape().elem_cnt()) {
      dst_blob->mut_dptr<float16>()[i] += src_blob_i->dptr<float16>()[i];
    }
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

  template<typename DptrT, DptrT* (*GetDptrT)(Blob*)>
  std::tuple<int64_t, DptrT*> CalcContinuousElemNumStartFrom(int64_t idx) const {
    std::tuple<int64_t, DptrT*> ret(0, nullptr);
    int64_t seg_idx = idx / elem_sum_.back();
    int64_t idx_in_seg = idx % elem_sum_.back();
    auto elem_sum_it = std::upper_bound(elem_sum_.begin(), elem_sum_.end(), idx_in_seg);
    CHECK(elem_sum_it != elem_sum_.end());
    std::get<0>(ret) = *elem_sum_it - idx_in_seg;
    int64_t bn_idx = elem_sum_it - elem_sum_.begin();
    int64_t idx_in_blob = idx_in_seg;
    if (bn_idx > 0) { idx_in_blob -= elem_sum_[bn_idx - 1]; }
    Blob* blob = BnInOp2Blob_(bns_->Get(bn_idx));
    std::get<1>(ret) = GetDptrT(blob)
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

static const char* GetConstDptr(Blob* blob) { return blob->dptr<char>(); }
static char* GetMutDptr(Blob* blob) { return blob->mut_dptr<char>(); }

void ConcatSplitPartDataContent(DeviceCtx* ctx, const DataContentDesc& in_desc,
                                const DataContentDesc& out_desc, int32_t part_id,
                                int32_t part_num) {
  size_t one_elem_size = in_desc.OneElemSize();
  BalancedSplitter bs(in_desc.TotalElemNum(), part_num);
  Range range = bs.At(part_id);
  int64_t in_idx = range.begin();
  int64_t in_elem_num = 0;
  const char* in_ptr = nullptr;
  int64_t out_idx = range.begin();
  int64_t out_elem_num = 0;
  char* out_ptr = nullptr;

  while (in_elem_num > 0 || out_elem_num > 0 || in_idx < range.end() || out_idx < range.end()) {
    if (in_elem_num == 0) {
      std::tie(in_elem_num, in_ptr) =
          in_desc.CalcContinuousElemNumStartFrom<const char, GetConstDptr>(in_idx);
      in_elem_num = std::min(in_elem_num, range.end() - in_idx);
      if (in_elem_num == 0) { break; }
      in_idx += in_elem_num;
    }
    if (out_elem_num == 0) {
      std::tie(out_elem_num, out_ptr) =
          out_desc.CalcContinuousElemNumStartFrom<char, GetMutDptr>(out_idx);
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
  part_num = std::min(part_num, Global<ThreadPool>::Get()->thread_num());
  if (part_num >= 2) {
    BlockingCounter bc(part_num);
    FOR_RANGE(int32_t, part_id, 0, part_num) {
      Global<ThreadPool>::Get()->AddWork([&ctx, &in_desc, &out_desc, part_id, &part_num, &bc]() {
        ConcatSplitPartDataContent(ctx, in_desc, out_desc, part_id, part_num);
        bc.Decrease();
      });
    }
    bc.WaitUntilCntEqualZero();
  } else {
    ConcatSplitPartDataContent(ctx, in_desc, out_desc, 0, 1);
  }
}

}  // namespace

template<typename T>
void BoxingKernel<T>::VirtualKernelInit() {
  const std::string& ibn_0 = op_attribute().input_bns(0);
  const std::string& obn_0 = op_attribute().output_bns(0);
  ibn_0_ = ConstructPbRpf(ibn_0);
  obn_0_ = ConstructPbRpf(obn_0);
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

ADD_CPU_DEFAULT_KERNEL_CREATOR(OperatorConf::kBoxingConf, BoxingKernel,
                               ARITHMETIC_DATA_TYPE_SEQ FLOAT16_DATA_TYPE_SEQ);

}  // namespace oneflow
