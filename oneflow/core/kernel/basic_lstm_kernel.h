#ifndef ONEFLOW_CORE_KERNEL_BASIC_LSTM_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_BASIC_LSTM_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow{
				
template<DeviceType device_type, typename T>
class BasicLstmKernel : public KernelIf<device_type> {
 public:
	OF_DISALLOW_COPY_AND_MOVE(BasicLstmKernel);
	BasicLstmKernel() = default;
	~BasicLstmKernel() = default;

 private:
  const PbMessage& GetBasicLstmOpConf() const = 0;
	bool HasInitHiddenInitializer() const = 0;
  bool HasInitCellInitializer() const = 0;
	bool NeedExternalH0() const;
	bool NeedExternalC0() const;
  Blob* GetHiddenBlob(std::function<Blob*(const std::string&)>) const;
	Blob* GetHiddenDiffBlob(std::function<Blob*(const std::string&)>) const;
	Blob* GetCellBlob(std::function<Blob*(const std::string&)>) const;
	Blob* GetCellDiffBlob(std::function<Blob*(const std::string&)>) const;

	void ForwardColNum(const KernelCtx&,
										std::function<Blob*(const std::string&)>) const override;
	void ForwardDataId(const KernelCtx&,
										std::function<Blob*(const std::string&)>) const override;
	void BackwardColNum(const KernelCtx&,
										std::function<Blob*(const std::string&)>) const override;
	void ForwardDataContent(const KernelCtx&, 
										std::function<Blob*(const std::string&)>) const override;
	void BackwardDataContent(const KenerCtx&,
										std::function<Blob*(const std::string&)>) const override;
	void InitPureModelTmpBlobs(
										DeviceCtx*,
										std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
	void InitModelBlobsWithDir(
			DeviceCtx*, int32_t part_id, int32_t part_num,
			const std::string& model_load_dir,
			std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  void InitModelBlobsWithRandomSeed(
			DeviceCtx*, std::mt19937* random_seed_gen,
			std::function<Blob*(const std::string&)>) const{}
	void VirtualKernelInit(const ParallelContext*) override;

	private:
	 bool need_external_h0_;
	 bool need_external_c0_;
};

template<DeviceType device_type, typename T>
struct BasicLstmKernelUtil {
	static void ComputeTanHDiff(DeviceCtx* ctx, int64_t n, const T* out,
															const T* out_diff, const T* rec_out_diff,
															T* plus_out_diff);
	static void ComputeSigmoidDiff(DeviceCtx* ctx, int64_t n, const T* out,
																const T* out_diff, const T* rec_out_diff,
																T* plus_op_out_diff);
};
								
}  //  namespace oneflow

#endif  //  ONEFLOW_CORE_KERNEL_BASIC_LSTM_KERNEL_H_
