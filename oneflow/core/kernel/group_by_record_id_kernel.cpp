#include "oneflow/core/kernel/group_by_record_id_kernel.h"

namespace oneflow {

namespace {

void InDim0ToOutDim0Transform(
    const Blob* in, const Blob* out,
    const std::function<void(int64_t in_dim0_idx, int64_t out_dim0_idx)>& Handler) {
  CHECK_GE(in->shape().NumAxes(), 1);
  CHECK_EQ(in->shape().NumAxes() + 1, out->shape().NumAxes());
  CHECK(in->has_record_id_in_device_piece_field());
  const int64_t out_dim0_size = out->shape().At(0);
  FOR_RANGE(int64_t, in_dim0_idx, 0, in->shape().At(0)) {
    const int64_t out_dim0_idx = in->record_id_in_device_piece(in_dim0_idx);
    CHECK_LT(out_dim0_idx, out_dim0_size);
    Handler(in_dim0_idx, out_dim0_idx);
  }
}

void InDim0ToOutDim0Dim1Transform(
    const Blob* in, const Blob* out,
    const std::function<void(int64_t in_dim0_idx, int64_t out_dim0_idx, int64_t out_dim1_idx)>&
        Handler) {
  std::vector<int64_t> cnt(static_cast<size_t>(out->shape().At(0)));
  InDim0ToOutDim0Transform(in, out, [&](int64_t in_dim0_idx, int64_t out_dim0_idx) {
    CHECK_LT(cnt[out_dim0_idx], out->shape().At(1));
    Handler(in_dim0_idx, out_dim0_idx, cnt[out_dim0_idx]);
    cnt[out_dim0_idx] += 1;
  });
}

}  // namespace

template<typename T>
void GroupByRecordIdKernel<T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in = BnInOp2Blob("in");
  Blob* out = BnInOp2Blob("out");
  CHECK_GE(in->shape().NumAxes(), 1);
  CHECK_EQ(in->shape().NumAxes() + 1, out->shape().NumAxes());
  const size_t one_row_size = in->shape().Count(1) * sizeof(T);
  CHECK_EQ(one_row_size, out->shape().Count(2) * sizeof(T));
  InDim0ToOutDim0Dim1Transform(
      in, out, [&](int64_t in_dim0_idx, int64_t out_dim0_idx, int64_t out_dim1_idx) {
        Memcpy<DeviceType::kCPU>(ctx.device_ctx, out->mut_dptr<T>(out_dim0_idx, out_dim1_idx),
                                 in->dptr<T>(in_dim0_idx), one_row_size);
      });
}

template<typename T>
void GroupByRecordIdKernel<T>::ForwardDim1ValidNum(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in = BnInOp2Blob("in");
  Blob* out = BnInOp2Blob("out");
  std::vector<int64_t> out_dim1_valid_num(static_cast<size_t>(out->shape().At(0)));
  InDim0ToOutDim0Transform(in, out, [&](int64_t in_dim0_idx, int64_t out_dim0_idx) {
    out_dim1_valid_num[out_dim0_idx] += 1;
  });
  FOR_RANGE(int64_t, out_dim0_idx, 0, out->shape().At(0)) {
    out->set_dim1_valid_num(out_dim0_idx, out_dim1_valid_num[out_dim0_idx]);
  }
}

template<typename T>
void GroupByRecordIdKernel<T>::ForwardDim2ValidNum(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in = BnInOp2Blob("in");
  Blob* out = BnInOp2Blob("out");
  CHECK(in->has_dim1_valid_num_field());
  InDim0ToOutDim0Dim1Transform(
      in, out, [&](int64_t in_dim0_idx, int64_t out_dim0_idx, int64_t out_dim1_idx) {
        out->set_dim2_valid_num(out_dim0_idx, out_dim1_idx, in->dim1_valid_num(in_dim0_idx));
      });
};

ADD_CPU_DEFAULT_KERNEL_CREATOR(OperatorConf::kGroupByRecordIdConf, GroupByRecordIdKernel,
                               POD_DATA_TYPE_SEQ);

}  // namespace oneflow
