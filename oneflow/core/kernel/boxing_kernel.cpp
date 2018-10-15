#include "oneflow/core/kernel/boxing_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/operator/op_conf.pb.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/thread/thread_manager.h"
#include "oneflow/core/common/data_type.h"

namespace oneflow {

void BoxingKernelUtil::CopyFromFirstToOtherBlobs(
    DeviceCtx* ctx, std::function<Blob*(const std::string&)> BnInOp2Blob,
    const PbRpf<std::string>& bns, CopyBlobFieldMthd Copy) {
  const Blob* blob_0 = BnInOp2Blob(bns.Get(0));
  FOR_RANGE(size_t, i, 1, bns.size()) { (BnInOp2Blob(bns.Get(i))->*Copy)(ctx, blob_0); }
}

PbRpf<std::string> BoxingKernelUtil::ConstructPbRpf(const std::string& s) {
  PbRpf<std::string> ret;
  ret.Reserve(1);
  ret.Add()->assign(s);
  return ret;
}

namespace {

template<typename Iter>
void ConcatSplitField(DeviceCtx* ctx, std::function<Blob*(const std::string&)> BnInOp2Blob,
                      const PbRpf<std::string>& concat_bns, int32_t concat_axis,
                      const PbRpf<std::string>& split_bns, int32_t split_axis) {
  Iter concat_it(BnInOp2Blob, &concat_bns, concat_axis);
  Iter split_it(BnInOp2Blob, &split_bns, split_axis);
  CopyFromIterToIter<DeviceType::kCPU>(ctx, concat_it, split_it);
  if (split_axis != 0) {
    BoxingKernelUtil::CopyFromFirstToOtherBlobs(ctx, BnInOp2Blob, split_bns,
                                                Iter::GetCopyBlobFieldMthd());
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
void BoxingKernel<T>::VirtualKernelInit(const ParallelContext*) {
  const std::string& ibn_0 = InputBns().Get(0);
  const std::string& obn_0 = OutputBns().Get(0);
  ibn_0_ = BoxingKernelUtil::ConstructPbRpf(ibn_0);
  obn_0_ = BoxingKernelUtil::ConstructPbRpf(obn_0);
  CHECK_EQ(kernel_conf().need_do_opaque_header(), false);
}

template<typename T>
template<typename Iter>
void BoxingKernel<T>::ForwardField(const KernelCtx& ctx,
                                   std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  if (boxing_conf().in_box_case() == BoxingOpConf::kConcatBox) {
    if (boxing_conf().out_box_case() == BoxingOpConf::kSplitBox) {
      ConcatSplitField<Iter>(ctx.device_ctx, BnInOp2Blob, InputBns(),
                             boxing_conf().concat_box().axis(), OutputBns(),
                             boxing_conf().split_box().axis());
    } else if (boxing_conf().out_box_case() == BoxingOpConf::kCloneBox) {
      ConcatSplitField<Iter>(ctx.device_ctx, BnInOp2Blob, InputBns(),
                             boxing_conf().concat_box().axis(), obn_0_, 0);
      BoxingKernelUtil::CopyFromFirstToOtherBlobs(ctx.device_ctx, BnInOp2Blob, OutputBns(),
                                                  Iter::GetCopyBlobFieldMthd());
    } else {
      UNIMPLEMENTED();
    }
  } else if (boxing_conf().in_box_case() == BoxingOpConf::kAddBox) {
    if (boxing_conf().out_box_case() == BoxingOpConf::kSplitBox) {
      ConcatSplitField<Iter>(ctx.device_ctx, BnInOp2Blob, ibn_0_, 0, OutputBns(),
                             boxing_conf().split_box().axis());
    } else if (boxing_conf().out_box_case() == BoxingOpConf::kCloneBox) {
      CopyField(ctx.device_ctx, BnInOp2Blob, BnInOp2Blob(ibn_0_.Get(0)), OutputBns(),
                Iter::GetCopyBlobFieldMthd());
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
  ForwardField<Dim0ValidNumIterator>(ctx, BnInOp2Blob);
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
void BoxingKernel<T>::SetColId(const KernelCtx& ctx,
                               std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  if (boxing_conf().in_box_case() == BoxingOpConf::kConcatBox
      && boxing_conf().concat_box().axis() == 0) {
    if (boxing_conf().out_box_case() == BoxingOpConf::kSplitBox
        && boxing_conf().split_box().axis() == 0) {
      ConcatSplitColId(BnInOp2Blob, InputBns(), OutputBns());
    } else {
      SetBlobsColId(BnInOp2Blob, OutputBns(), MaxColIdInBlobs(BnInOp2Blob, InputBns()));
    }
  } else {
    SetBlobsColId(BnInOp2Blob, OutputBns(), BnInOp2Blob(InputBns().Get(0))->col_id());
  }
}

template<typename T>
void BoxingKernel<T>::SetMaxColId(const KernelCtx& ctx,
                                  std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  for (const std::string& obn : OutputBns()) {
    int32_t max_col_num_in_blob = 0;
    Blob* out_blob = BnInOp2Blob(obn);
    FOR_RANGE(int32_t, i, 0, out_blob->static_shape().At(0)) {
      max_col_num_in_blob = std::max(max_col_num_in_blob, out_blob->col_num(i));
    }
    out_blob->set_max_col_id(max_col_num_in_blob - 1);
  }
}

#define INIT_BOXING_KERNEL(type_cpp, type_proto) template class BoxingKernel<type_cpp>;
OF_PP_FOR_EACH_TUPLE(INIT_BOXING_KERNEL, ARITHMETIC_DATA_TYPE_SEQ PB_LIST_DATA_TYPE_SEQ);

}  // namespace oneflow
