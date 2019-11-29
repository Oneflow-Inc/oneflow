#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/common/gdb.h"

namespace oneflow {

namespace {

void CheckSameRecordIdInDevicePiece(const PbRpf<std::string>& bns,
                                    const std::function<Blob*(const std::string&)>& BnInOp2Blob) {
  const void* mem_ptr = BnInOp2Blob(bns.Get(0))->record_id_in_device_piece_ptr();
  size_t len = BnInOp2Blob(bns.Get(0))->ByteSizeOfRecordIdInDevicePieceField();
  FOR_RANGE(int, i, 1, bns.size()) {
    CHECK_EQ(std::memcmp(BnInOp2Blob(bns.Get(i))->record_id_in_device_piece_ptr(), mem_ptr, len),
             0);
  }
}

void ClearBlobDim0ValidNumIfNeed(const PbRpf<std::string>& bns,
                                 const std::function<Blob*(const std::string&)>& BnInOp2Blob) {
  for (const auto& bn : bns) {
    Blob* blob = BnInOp2Blob(bn);
    if (blob != nullptr && blob->has_dim0_valid_num_field()) {
      std::memset(blob->mut_dim0_valid_num_ptr(), 0, blob->ByteSizeOfDim0ValidNumField());
    }
  }
}

}  // namespace

void Kernel::Init(const JobDesc* job_desc, const KernelConf& kernel_conf, DeviceCtx* device_ctx) {
  job_desc_ = job_desc;
  kernel_conf_ = kernel_conf;
  VirtualKernelInit(device_ctx);
}

void Kernel::InitModelAndConstBuf(const KernelCtx& ctx,
                                  std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  InitConstBufBlobs(ctx.device_ctx, BnInOp2Blob);
}

void Kernel::Launch(const KernelCtx& ctx,
                    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  if (kernel_conf_.is_forward()) {
    gdb::ForwardEnterBreakPoint(op_attribute(), BnInOp2Blob);
    Forward(ctx, BnInOp2Blob);
    gdb::ForwardLeaveBreakPoint(op_attribute(), BnInOp2Blob);
  } else {
    UNIMPLEMENTED();
  }
}

const LogicalBlobId& Kernel::BnInOp2Lbi(const std::string& bn_in_op) const {
  return op_attribute().bn_in_op2lbi().at(bn_in_op);
}

bool Kernel::HasEmptyShapeBlob(const PbRpf<std::string>& bns,
                               const std::function<Blob*(const std::string&)>& BnInOp2Blob) const {
  for (const auto& bn : bns) {
    Blob* blob = BnInOp2Blob(bn);
    if (blob && blob->IsShapeEmpty()) { return true; }
  }
  return false;
}

void Kernel::CheckSameDim0ValidNum(
    const PbRpf<std::string>& bns,
    const std::function<Blob*(const std::string&)>& BnInOp2Blob) const {
  const Blob* first_blob = nullptr;
  FOR_RANGE(int, i, 0, bns.size()) {
    const Blob* cur_blob = BnInOp2Blob(bns.Get(i));
    if (cur_blob == nullptr) { continue; }
    first_blob = cur_blob;
    break;
  }
  const void* mem_ptr = first_blob->dim0_valid_num_ptr();
  size_t len = first_blob->ByteSizeOfDim0ValidNumField();
  FOR_RANGE(int, i, 1, bns.size()) {
    const Blob* cur_blob = BnInOp2Blob(bns.Get(i));
    if (cur_blob == nullptr) { continue; }
    CHECK_EQ(std::memcmp(cur_blob->dim0_valid_num_ptr(), mem_ptr, len), 0);
  }
}

void Kernel::Forward(const KernelCtx& ctx,
                     std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  if (kernel_conf_.need_do_dim0_valid_num()) {
    CHECK(!kernel_conf_.need_do_opaque_header());
    ForwardDim0ValidNum(ctx, BnInOp2Blob);
  }
  if (HasEmptyShapeBlob(op_attribute().input_bns(), BnInOp2Blob) && !NeedForwardIfBlobEmpty()) {
    ClearBlobDim0ValidNumIfNeed(op_attribute().output_bns(), BnInOp2Blob);
    return;
  }
  if (kernel_conf_.need_do_dim1_valid_num()) {
    CHECK(!kernel_conf_.need_do_opaque_header());
    ForwardDim1ValidNum(ctx, BnInOp2Blob);
  }
  if (kernel_conf_.need_do_dim2_valid_num()) {
    CHECK(!kernel_conf_.need_do_opaque_header());
    ForwardDim2ValidNum(ctx, BnInOp2Blob);
  }
  if (kernel_conf_.need_do_record_id_in_device_piece()) {
    CHECK(!kernel_conf_.need_do_opaque_header());
    ForwardRecordIdInDevicePiece(ctx, BnInOp2Blob);
  }
  ForwardDataContent(ctx, BnInOp2Blob);
  if (kernel_conf_.need_do_opaque_header()) {
    ForwardPackedHeader(ctx, BnInOp2Blob);
  } else {
    if (kernel_conf_.need_do_data_id()) { ForwardDataId(ctx, BnInOp2Blob); }
    if (kernel_conf_.need_do_col_num()) { ForwardColNum(ctx, BnInOp2Blob); }
  }
}

template<DeviceType device_type>
void KernelIf<device_type>::ForwardDataId(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  CopyField(ctx.device_ctx, BnInOp2Blob, op_attribute().input_bns(), op_attribute().output_bns(),
            &Blob::CopyDataIdFrom);
}

template<DeviceType device_type>
void KernelIf<device_type>::ForwardColNum(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  CopyField(ctx.device_ctx, BnInOp2Blob, op_attribute().input_bns(), op_attribute().output_bns(),
            &Blob::CopyColNumFrom);
}

template<DeviceType device_type>
void KernelIf<device_type>::ForwardDim0ValidNum(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  CHECK(kernel_conf().can_naive_do_dim0_valid_num());
  CheckSameDim0ValidNum(op_attribute().input_bns(), BnInOp2Blob);
  CopyField(ctx.device_ctx, BnInOp2Blob, BnInOp2Blob(op_attribute().input_bns(0)),
            op_attribute().output_bns(), &Blob::CopyDim0ValidNumFrom);
}

template<DeviceType device_type>
void KernelIf<device_type>::ForwardRecordIdInDevicePiece(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  CHECK(kernel_conf().can_naive_do_record_id_in_device_piece());
  CheckSameRecordIdInDevicePiece(op_attribute().input_bns(), BnInOp2Blob);
  CopyField(ctx.device_ctx, BnInOp2Blob, BnInOp2Blob(op_attribute().input_bns(0)),
            op_attribute().output_bns(), &Blob::CopyRecordIdInDevicePieceFrom);
}

template<DeviceType device_type>
void KernelIf<device_type>::ForwardPackedHeader(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  CopyField(ctx.device_ctx, BnInOp2Blob, op_attribute().input_bns(), op_attribute().output_bns(),
            &Blob::CopyHeaderFrom);
}

template<DeviceType device_type>
void KernelIf<device_type>::CopyField(DeviceCtx* ctx,
                                      std::function<Blob*(const std::string&)> BnInOp2Blob,
                                      const Blob* from_blob, const PbRpf<std::string>& to_bns,
                                      void (Blob::*Copy)(DeviceCtx*, const Blob*)) const {
  for (const std::string& to_bn : to_bns) { (BnInOp2Blob(to_bn)->*Copy)(ctx, from_blob); }
}

template<DeviceType device_type>
void KernelIf<device_type>::CopyField(DeviceCtx* ctx,
                                      std::function<Blob*(const std::string&)> BnInOp2Blob,
                                      const PbRpf<std::string>& from_bns,
                                      const PbRpf<std::string>& to_bns,
                                      void (Blob::*Copy)(DeviceCtx*, const Blob*)) const {
  if (from_bns.size() == 1) {
    const Blob* in_blob = BnInOp2Blob(from_bns[0]);
    CopyField(ctx, BnInOp2Blob, in_blob, to_bns, Copy);
  } else if (to_bns.size() == 1) {
    Blob* in_blob = BnInOp2Blob(from_bns[0]);
    Blob* out_blob = BnInOp2Blob(to_bns[0]);
    (out_blob->*Copy)(ctx, in_blob);
  } else {
    CHECK_EQ(from_bns.size(), to_bns.size());
    FOR_RANGE(size_t, i, 0, from_bns.size()) {
      Blob* in_blob = BnInOp2Blob(from_bns[i]);
      Blob* out_blob = BnInOp2Blob(to_bns[i]);
      (out_blob->*Copy)(ctx, in_blob);
    }
  }
}

std::unique_ptr<const Kernel> ConstructKernel(const JobDesc* job_desc, const KernelConf& conf,
                                              DeviceCtx* device_ctx) {
  auto op_type = conf.op_attribute().op_conf().op_type_case();
  Kernel* rptr = kernel_registration::CreateKernel(conf);
  if (rptr == nullptr) { rptr = NewObj<Kernel>(op_type, conf); }
  CHECK_NOTNULL(rptr);
  rptr->Init(job_desc, conf, device_ctx);
  return std::unique_ptr<const Kernel>(rptr);
}

#define INSTANTIATE_KERNEL_IF(device_type) template class KernelIf<device_type>;

OF_PP_FOR_EACH_TUPLE(INSTANTIATE_KERNEL_IF, DEVICE_TYPE_SEQ);

}  // namespace oneflow
