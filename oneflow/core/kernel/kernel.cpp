#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/common/gdb.h"
#include "oneflow/core/common/cached_caller.h"
#include "oneflow/core/kernel/runtime_blob_shape_infer_helper.h"

namespace oneflow {

namespace {

bool IsAllBlobEmpty(const PbRpf<std::string>& bns,
                    const std::function<Blob*(const std::string&)>& BnInOp2Blob) {
  for (const auto& bn : bns) {
    Blob* blob = BnInOp2Blob(bn);
    if (blob && !blob->IsBodyEmpty()) { return false; }
  }
  return true;
}

}  // namespace

Kernel::~Kernel() {
  if (shape_infer_helper_ != nullptr) { delete shape_infer_helper_; }
}

void Kernel::Init(const JobDesc* job_desc, const KernelConf& kernel_conf, DeviceCtx* device_ctx) {
  job_desc_ = job_desc;
  kernel_conf_ = kernel_conf;
  shape_infer_helper_ =
      new RuntimeBlobShapeInferHelper(this->op_conf(), this->kernel_conf(), &this->job_desc());
  VirtualKernelInit(device_ctx);
}

void Kernel::InitModelAndConstBuf(const KernelCtx& ctx,
                                  std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  InitConstBufBlobs(ctx.device_ctx, BnInOp2Blob);
}

void Kernel::Launch(const KernelCtx& ctx,
                    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  auto CachedBnInOp2Blob = WithResultCached(BnInOp2Blob);
  gdb::ForwardEnterBreakPoint(op_attribute(), CachedBnInOp2Blob);
  Forward(ctx, CachedBnInOp2Blob);
  gdb::ForwardLeaveBreakPoint(op_attribute(), CachedBnInOp2Blob);
}

const LogicalBlobId& Kernel::BnInOp2Lbi(const std::string& bn_in_op) const {
  return op_attribute().bn_in_op2lbi().at(bn_in_op);
}

void Kernel::CheckSameDim0ValidNum(
    const PbRpf<std::string>& bns,
    const std::function<Blob*(const std::string&)>& BnInOp2Blob) const {
  UNIMPLEMENTED();
}

void Kernel::Forward(const KernelCtx& ctx,
                     std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  ForwardHeader(ctx, BnInOp2Blob);
  if (IsAllBlobEmpty(op_attribute().output_bns(), BnInOp2Blob) && IsStateless()) { return; }
  ForwardDataContent(ctx, BnInOp2Blob);
}

void Kernel::ForwardHeader(const KernelCtx& ctx,
                           std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  if (kernel_conf_.need_do_opaque_header()) {
    ForwardPackedHeader(ctx, BnInOp2Blob);
  } else {
    if (kernel_conf_.need_do_dense_shape()) { ForwardDenseShape(ctx, BnInOp2Blob); }
    if (kernel_conf_.need_do_lod()) { ForwardLoD(ctx, BnInOp2Blob); }
  }
}

void Kernel::ForwardLoD(const KernelCtx& ctx,
                        std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  UNIMPLEMENTED();
}

void Kernel::ForwardDenseShape(const KernelCtx& ctx,
                               std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  return shape_infer_helper_->InferDenseShape(BnInOp2Blob);
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
