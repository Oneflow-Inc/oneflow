#ifndef ONEFLOW_CORE_KERNEL_BOXING_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_BOXING_KERNEL_H_

#include <string>
#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/kernel_manager.h"
#include "oneflow/core/kernel/kernel_context.h"
#include "oneflow/core/actor/cuda_device_context.h"

namespace oneflow {
  
template<DeviceType device_type, typename floating_point_type> 
class BoxingKernel; 

template<typename floating_point_type>
class BoxingKernel<DeviceType::kALL, floating_point_type> : 
  public Kernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BoxingKernel);
  virtual ~BoxingKernel() = default;

  void InitFromOpProto(const OperatorProto& op_proto) override;

  void Forward(const KernelCtx&,
               std::function<Blob*(const std::string&)>) const override;
  void Backward(const KernelCtx&,
                std::function<Blob*(const std::string&)>) const override;
 protected:
  BoxingKernel() = default;
  // Mark: seems inappropriate to use ctx into OF* fucntions. However, since 
  // it is a virtual function, we do not know whether it will be excecuted on
  // cpu or gpu yet.
  virtual void Memcpy(const KernelCtx& ctx,
      void* dst, const void* src, size_t sz) = 0;
  virtual void BlobCpy(const KernelCtx& ctx, const Blob* a, Blob* b) = 0;
  virtual void BlobAdd(const KernelCtx& ctx, const Blob*, Blob*) = 0;
  virtual void BlasAxpy(const KernelCtx& ctx, const int N,
      const floating_point_type alpha,
      const floating_point_type* X, const int incX,
      floating_point_type *Y, const int incY) = 0;
  virtual void BlasScal(const KernelCtx& ctx, const int n,
      const floating_point_type alpha, floating_point_type* x, int incx) = 0;

  // For concat ==> (split/clone) box:
  // a copy_rule means a step of memory action during runtime. Since the 
  // blob shapes are fixed after initilization, the offsets of each blobs 
  // are therefore constant during execution. We can write down the offsets
  // of each blob in the first time, and directly use these records to do
  // memory-copy in the next running.
  struct copy_rule {
    std::string src_bn, dst_bn; /* Blob names*/
    uint64_t src_offset, dst_offset; /* corresponding offsets*/
    uint64_t copy_sz; 
  };
  void InferCopyRules(std::function<Blob*(const std::string&)>) const;
  void ConstructCopyRulesFromSlice(
      const std::map<const std::string*, int64_t>& src_bn2slice, 
      const std::map<const std::string*, int64_t>& dst_bn2slice,
      int64_t seg_cnt, int64_t slice_sz, int concat_axis, 
      std::vector<struct copy_rule>& rules) const; 

  void ConstructFwCloneRules(std::function<Blob*(const std::string&)>) const;
  void InferCopyRulesFromBns(
    std::function<Blob*(const std::string&)> BnInOp2BlobPtr,
    const std::vector<std::string>& src_bns,
    const std::vector<std::string>& dst_bns,
    std::vector<copy_rule>& copy_rules
    ) const;

  // Forward function for Add box
  void AddBoxForward(
      const KernelCtx& ctx,
      std::function<Blob*(const std::string&)>) const;
  // Backward function for Add box
  void AddBoxBackward(
      const KernelCtx& ctx,
      std::function<Blob*(const std::string&)>) const;

  // Forward function for Concat box 
  void ConcatBoxForward(
      const KernelCtx& ctx,
      std::function<Blob*(const std::string&)>) const;

  // Backward function for Concat-split box && concat-clone box
  void ConcatBoxBackward(
      const KernelCtx& ctx,
      std::function<Blob*(const std::string&)>) const;

  using ExecFunc = void (BoxingKernel<DeviceType::kALL,
      floating_point_type>::*) (const KernelCtx&,
        std::function<Blob*(const std::string&)>) const;

  mutable std::vector<copy_rule> fw_copy_rules;
  mutable std::vector<copy_rule> bw_copy_rules;
  ExecFunc fw_func_;
  ExecFunc bw_func_;
};

template<typename floating_point_type>
class BoxingKernel<DeviceType::kCPU, floating_point_type> final : 
  public BoxingKernel<DeviceType::kALL, floating_point_type>  {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BoxingKernel);
  BoxingKernel() = default;
  ~BoxingKernel() = default;

  // implementations of math virtual functions on cpu
  void Memcpy(const KernelCtx& ctx,
      void* dst, const void* src, size_t sz) override;
  void BlobCpy(const KernelCtx& ctx, const Blob* a, Blob* b) override;
  void BlobAdd(const KernelCtx& ctx, const Blob*, Blob*) override;
  void BlasAxpy(const KernelCtx& ctx, const int N,
      const floating_point_type alpha,
      const floating_point_type* X, const int incX, floating_point_type *Y,
      const int incY) override;
  void BlasScal(const KernelCtx& ctx, const int n,
      const floating_point_type alpha, floating_point_type* x,
      int incx) override;
};

template<typename floating_point_type>
class BoxingKernel<DeviceType::kGPU, floating_point_type> final : 
  public BoxingKernel<DeviceType::kALL, floating_point_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BoxingKernel);
  BoxingKernel() = default;
  ~BoxingKernel() = default;

  // implementations of math virtual functions on gpu
  void Memcpy(const KernelCtx& ctx,
      void* dst, const void* src, size_t sz) override;
  void BlobCpy(const KernelCtx& ctx, const Blob* a, Blob* b) override;
  void BlobAdd(const KernelCtx& ctx, const Blob*, Blob*) override;
  void BlasAxpy(const KernelCtx& ctx, const int N,
      const floating_point_type alpha,
      const floating_point_type* X, const int incX,
      floating_point_type *Y, const int incY) override;
  void BlasScal(const KernelCtx& ctx, const int n,
      const floating_point_type alpha, floating_point_type* x,
      int incx) override;
};

}  // namespace oneflow

#endif // ONEFLOW_CORE_KERNEL_BOXING_KERNEL_H_
