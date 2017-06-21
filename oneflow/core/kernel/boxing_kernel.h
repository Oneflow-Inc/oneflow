#ifndef ONEFLOW_CORE_KERNEL_BOXING_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_BOXING_KERNEL_H_

#include <string>
#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/kernel_manager.h"
#include "oneflow/core/kernel/kernel_context.h"
#include "oneflow/core/actor/cuda_device_context.h"

namespace oneflow {
  
template<DeviceType device_type, typename floating_point_type>
class BoxingKernel : public Kernel {

};

template<typename floating_point_type>
class BoxingKernel<DeviceType::kALL, floating_point_type> : 
  public Kernel {
 public:
  //OF_DISALLOW_COPY_AND_MOVE(BoxingKernel);
  BoxingKernel() = default;
  ~BoxingKernel() = default;

  void InitFromOpProto(const OperatorProto& op_proto);

  void Forward(const KernelCtx&,
               std::function<Blob*(const std::string&)>) const override;
  void Backward(const KernelCtx&,
                std::function<Blob*(const std::string&)>) const override;
 protected:
  virtual void OFMemcpy(const KernelCtx& ctx, \
      void* dst, const void* src, size_t sz) = 0;
  virtual void OFAddBlob(const KernelCtx& ctx, Blob*, Blob*)=0;
  virtual void OFBlasAxpy( 
    const KernelCtx& ctx, 
    const int N, const floating_point_type alpha, 
    const floating_point_type* X, const int incX, 
    floating_point_type *Y, const int incY
    ) = 0;
  virtual void OFBlasScal(
    const KernelCtx& ctx,
    const int n, const floating_point_type alpha,
    floating_point_type* x, int incx
    ) = 0;
 private:
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
  void InferCopyRules(std::function<Blob*(const std::string&)>);
  void ConstructRulesFromShape(
      std::map<const std::string*, int64_t>& src_bn2slice, 
      std::map<const std::string*, int64_t>& dst_bn2slice,
      int64_t seg_cnt, int64_t slice_sz, int concat_axis, 
      std::vector<struct copy_rules>& rules
  ); 

  void ConstructFwCloneRules(std::function<Blob*(const std::string&)>);
      
  void InferCopyRulesFromBns(
    std::function<Blob*(const std::string&)> BnInOp2BlobPtr,
    std::vector<std::string>& src_bns,
    std::vector<std::string>& dst_bns,
    std::vector<copy_rule>& copy_rules
    );

  void AddBoxForward(
      const KernelCtx& ctx,
      std::function<Blob*(const std::string&)>) const;

  void AddBoxBackward(
      const KernelCtx& ctx,
      std::function<Blob*(const std::string&)>) const;

  void ConcatBoxForward(
      const KernelCtx& ctx,
      std::function<Blob*(const std::string&)>) const;
  void ConcatBoxBackward(
      const KernelCtx& ctx,
      std::function<Blob*(const std::string&)>) const;

  using ExecFunc = void (BoxingKernel<DeviceType::kALL, \
      floating_point_type>::*) (
      const KernelCtx&, 
      std::function<Blob*(const std::string&)>
      ) const;

  std::vector<copy_rule> fw_copy_rules;
  std::vector<copy_rule> bw_copy_rules;
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

  virtual void OFMemcpy(const KernelCtx& ctx, \
      void* dst, const void* src, size_t sz);
  virtual void OFAddBlob(const KernelCtx& ctx, Blob*, Blob*);
  virtual void OFBlasAxpy( 
    const KernelCtx& ctx, 
    const int N, const floating_point_type alpha, 
    const floating_point_type* X, const int incX, 
    floating_point_type *Y, const int incY
    );
  virtual void OFBlasScal(
    const KernelCtx& ctx,
    const int n, const floating_point_type alpha,
    floating_point_type* x, int incx
    );
};

template<typename floating_point_type>
class BoxingKernel<DeviceType::kGPU, floating_point_type> final : 
  public BoxingKernel<DeviceType::kALL, floating_point_type> {
  public:
  OF_DISALLOW_COPY_AND_MOVE(BoxingKernel);
  BoxingKernel() = default;
  ~BoxingKernel() = default;

  virtual void OFMemcpy(const KernelCtx& ctx, \
      void* dst, const void* src, size_t sz);
  virtual void OFAddBlob(const KernelCtx& ctx, Blob*, Blob*);
  virtual void OFBlasAxpy( 
    const KernelCtx& ctx, 
    const int N, const floating_point_type alpha, 
    const floating_point_type* X, const int incX, 
    floating_point_type *Y, const int incY
    );
  virtual void OFBlasScal(
    const KernelCtx& ctx,
    const int n, const floating_point_type alpha,
    floating_point_type* x, int incx
    );
};

} // namespace oneflow

#endif // ONEFLOW_CORE_KERNEL_BOXING_KERNEL_H_
