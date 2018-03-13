#include "oneflow/core/kernel/basic_lstm_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void BasicLstmKernel<device_type, T>::VirtualKernelInit(
		const ParallelContext*) {
	auto& input_bns = this->kernel_conf().input_bns();
	need_external_h0_ = 
			std::find(input_bns.begin(), input_bns.end(), "h0") != input_bns.end();
	need_external_c0_ = 
			std::find(input_bns.begin(), input_bns.end(), "c0") != input_bns.end();
}

template<DeviceType device_type, typename T>
const PbMessage& BasicLstmKernel<device_type, T>::GetBasicLstmOpConf() const {
	return this->op_conf().basic_lstm_conf();
}

template<DeviceType device_type, typename T>
bool BasicLstmKernel<device_type, T>::HasInitHiddenInitializer() const {
	return this->op_conf().basic_lstm_conf().has_init_hidden_initializer();
}

template<DeviceType device_type, typename T>
bool BasicLstmKernel<device_type, T>::HasInitCellInitializer() const {
	return this->op.conf().basic_lstm_conf().has_init_cell_initializer();
}

template<DeviceType device_type, typename T>
bool BasicLstmKernel<device_type, T>::NeedExternalH0() const {
	return need_external_h0_;
}

template<DeviceType device_type, typename T>
bool BasicLstmKernel<device_type, T>::NeedExternalC0() const {
	return need_external_c0_;
}

template<DeviceType device_type, typename T>
Blob* BasicLstmKernel<device_type, T>::GetHiddenBlob(
		std::function<Blob*(const std::string&)> BnInOp2Blob) const {
	if (BnInOp2Blob("in")->col_id() == 0) { return BnInOp2Blob("h0"); }
	return BnInOp2Blob("rec_in");
}

template<DeviceType device_type, typename T>
Blob* BasicLstmKernel<device_type, T>::GetCellBlob(
		std::function<Blob*(const std::string&)> BnInOp2Blob) const {
	if (BnInOp2Blob("in")->col_id() == 0) { return BnInOp2Blob("c0"); }
	return BnInOp2Blob("cell_in");
}

template<DeviceType device_type, typename T>
Blob* BasicLstmKernel<device_type, T>::GetHiddenDiffBlob(
		std::function<Blob*(const std::string&)> BnInOp2Blob) const {
	if (BnInOp2Blob("in")->col_id() == 0) { return BnInOp2Blob("h0_diff"); }
	return BnInOp2Blob("rec_in_diff");
}

template<DeviceType device_type, typename T>
Blob* BasicLstmKernel<device_type, T>::GetCellDiffBlob(
		std::function<Blob*(const string&)> BnInOp2Blob) const {
	if (BnInOp2Blob("in")->col_id() == 0) { return BnInOp2Blob("c0_diff"); }
	return BnInOp2Blob("cell_in_diff");
}


template<DeviceType device_type, typename T>
void BasicLstmKernel<device_type, T>:ForwardDataId(
		const KernelCtx& ctx,
		std::function<Blob*(const std::string&)> BnInOp2Blob) const {
	BnInOp2Blob("out")->CopyDataIdFrom(ctx.device_ctx, BnInOp2Blob("in"));	
}

template<DeviceType device_type, typename T>
void BasicLstmKernel<device_type, T>::ForwardColNum(
		const KernelCtx& ctx,
		std::function<Blob*(const std::string&)> BnInOp2Blob) cosnt {
	BnInOp2Blob("out")->CopyColNumFrom(ctx.device_ctx, BnInOp2Blob("in"));
	BnInOp2Blob("rec_out")->CopyColNumFrom(ctx.device_ctx, BnInOp2Blob("in"));
	BnInOp2Blob("cell_out")->CopyColNumFrom(ctx.device_ctx, BnInOp2Blob("in"));
}

template<DeviceType device_type, typename T>
void BasicLstmKernel<device_type, T>::BackWardColNum(
		const KernelCtx& ctx,
		std::function<Blob*(const std::string&)> BnInOp2Blob) const {
	BnInOp2Blob("in_diff")
			->CopyColNumFrom(ctx.device_ctx, BnInOp2Blob("out_diff"));
	BnInOp2Blob("rec_in_diff")
			->CopyColNumFrom(ctx.device_ctx, BnInOp2Blob("out_diff"));
	BnInOp2Blob("cell_in_diff")
			->CopyColNumFrom(ctx.device_type, BnInOp2Blob("out_diff"));
}

template<DeviceType device_type, typename T>
void BasicLstmKernel<device_type, T>::ForwardDataContent(
		const KenelCtx& ctx,
		std::function<Blob*(const std::string&)> BnInOp2Blob) const {
	const Blob* hidden_blob = this->GetHiddenBlob(BnInOp2Blob);
	const Blob* cell_blob = this->GetCellBlob(BnInOp2Blob);
	Blob* f_gate_out_blob = BnInOp2Blob("f_gate_out");
	Blob* i_gate_out_blob = BnInOp2Blob("i_gate_out");
	Blob* o_gate_out_blob = BnInOp2Blob("o_gate_out");
	Blob* c_gate_out_blob = BnInOp2Blob("c_gate_out");
	Blob* out_blob = BnInOp2Blob("out");
	Blob* f_out_blob = BnInOp2Blob("f_out");
	Blob* i_out_blob = BnInOp2Blob("i_out");
	Blob* o_out_blob = BnInOp2Blob("o_out");
	Blob* c_out_blob = BnInOp2Blob("c_out");
	// f_gate_out = in * i2h_f_weight + hidden * h2h_f_weight +
	//							bias * bias_f_mulipler
	KenerUtil<device_type, T>::BlobGemm(ctx.device_ctx, CblasNoTrans, CblasTran,
																			static_cast<T>(1), static_cast<T>(0),
																			BnInOp2Blob("in"), BnInOp2Blob("i2h_f_weight"),
																			f_gate_out);
	KernelUtil<device_type, T>::BlobGemm(ctx.device_ctx, CblasNoTrans, CblasTrans,
																			 static_cast<T>(1), static_cast<T>(1),
																			 hidden_blob, BnInOp2Blob("h2h_f_weight"),
																			 f_gate_out);
  KernelUtil<device_type, T>::BlobGemm(ctx.device_ctx, CblasNoTrans, CblasNoTrans,
																			 static_cast<T>(1), static_cast<T>(1),
																			 BnInOp2Blob("bias_f_multiplier"), BnInOp2Blob("bias_f"),
																			 f_gate_out);
	// f_out = sigmoid(f_gate_out)    
	KernelUtil<device_type, T>::Sigmoid(ctx.device_ctx, out_blob->shape().elem_cnt(),
																			f_gate_out_blob->dptr<T>(), f_out_blob->mut_dptr<T>());

	// i_gate_out = in * i2h_i_weight + hidden * h2h_i_weight +
	//							bias_i * bias_i_mulipler
	KenerUtil<device_type, T>::BlobGemm(ctx.device_ctx, CblasNoTrans, CblasTran,
																			static_cast<T>(1), static_cast<T>(0),
																			BnInOp2Blob("in"), BnInOp2Blob("i2h_i_weight"),
																			i_gate_out);
	KernelUtil<device_type, T>::BlobGemm(ctx.device_ctx, CblasNoTrans, CblasTrans,
																			 static_cast<T>(1), static_cast<T>(1),
																			 hidden_blob, BnInOp2Blob("h2h_i_weight"),
																			 i_gate_out);
  KernelUtil<device_type, T>::BlobGemm(ctx.device_ctx, CblasNoTrans, CblasNoTrans,
																			 static_cast<T>(1), static_cast<T>(1),
																			 BnInOp2Blob("bias_i_multiplier"), BnInOp2Blob("bias_i"),
																			 i_gate_out);
	// i_out = sigmoid(i_gate_out)  
	KernelUtil<device_type, T>::Sigmoid(ctx.device_ctx, out_blob->shape().elem_cnt(),
																			i_gate_out_blob->dptr<T>(), i_out_blob->mut_dptr<T>());

	// c_gate_out = in * i2h_c_weight + hidden * h2h_c_weight +
	//							bias_c * bias_c_mulipler
	KenerUtil<device_type, T>::BlobGemm(ctx.device_ctx, CblasNoTrans, CblasTran,
																			static_cast<T>(1), static_cast<T>(0),
																			BnInOp2Blob("in"), BnInOp2Blob("i2h_c_weight"),
																			c_gate_out);
	KernelUtil<device_type, T>::BlobGemm(ctx.device_ctx, CblasNoTrans, CblasTrans,
																			 static_cast<T>(1), static_cast<T>(1),
																			 hidden_blob, BnInOp2Blob("h2h_c_weight"),
																			 c_gate_out);
  KernelUtil<device_type, T>::BlobGemm(ctx.device_ctx, CblasNoTrans, CblasNoTrans,
																			 static_cast<T>(1), static_cast<T>(1),
																			 BnInOp2Blob("bias_c_multiplier"), BnInOp2Blob("bias_c"),
																			 c_gate_out);

	// c_out = tanh(c_gate_out)     
	KernelUtil<device_type, T>::TanH(ctx.device_ctx, out_blob->shape().elem_cnt(),
																	 c_gate_out_blob->dptr<T>(), c_out_blob->mut_dptr<T>());


	// o_gate_out = in * i2h_o_weight + hidden * h2h_o_weight +
	//							bias_o * bias_o_mulipler
	KenerUtil<device_type, T>::BlobGemm(ctx.device_ctx, CblasNoTrans, CblasTran,
																			static_cast<T>(1), static_cast<T>(0),
																			BnInOp2Blob("in"), BnInOp2Blob("i2h_o_weight"),
																			o_gate_out);
	KernelUtil<device_type, T>::BlobGemm(ctx.device_ctx, CblasNoTrans, CblasTrans,
																			 static_cast<T>(1), static_cast<T>(1),
																			 hidden_blob, BnInOp2Blob("h2h_o_weight"),
																			 o_gate_out);
  KernelUtil<device_type, T>::BlobGemm(ctx.device_ctx, CblasNoTrans, CblasNoTrans,
																			 static_cast<T>(1), static_cast<T>(1),
																			 BnInOp2Blob("bias_o_multiplier"), BnInOp2Blob("bias_o"),
																			 o_gate_out);

	// o_out = sigmoid(o_gate_out)  
	KernelUtil<device_type, T>::Sigmoid(ctx.device_ctx, out_blob->shape().elem_cnt(),
																			o_gate_out_blob->dptr<T>(), o_out_blob->mut_dptr<T>());


	// rec_out = o_out *  tanh(c_out)  
  KernelUtil<device_type, T>::TanH(ctx.device_ctx, out_blob->shape().elem_cnt(),
																	 c_gate_out_blob->dptr<T>(), rec_out->mut_dptr<T>());
	KernelUtil<device_type, T>::Mul(ctx.device_ctx, out_blob->shape().elem_cnt(),
																	o_out_blob->dptr<T>(), rec_out_blob->dptr<T>(),
																	rec_out_blob->mut_dptr<T>());
	

	// cell_out = f_out * cell_in + i_out *  c_out
	KernelUtil<device_type, T>::Mul(ctx.device_type, out_blob->shape().elem_cnt(),
																	f_out_blob->dptr<T>(), cell_in_blob->dptr<T>(),
																	cell_out_blob->mut_dptr<T>());
	KernelUtil<device_type, T>::Mul(ctx.device_type, out_blob->shape().elem_cnt(),
																	i_out_blob->dptr<T>(), cell_out_blob->dptr<T>(),
																	updata_blob->mut_dptr<T>());
	KernelUtil<device_type, T>::Axpy(ctx.device_type, out_blob->shape().elem_cnt(),
																	static_cast<T>(1), cell_out_blob->dptr<T>(),
																	static_cast<T>(1), update_blob->dptr<T>(),
																	static_cast<T>(1));


template<DeviceType device_type, typename T>
void BasicLstmKernel<device_type, T>::BackwardDataContent(
		const KernelCtx& ctx,
		std::function<Blob*(const std:;string&)> BnInOp2Blob) const {
	const Blob* in_blob = BnInOp2Blob("in")
  const Blob* out_blob = BnInOp2Blob("out") 
	const Blob* hidden_blob = this->GetHiddenBlob(BnInOp2Blob);
	const Blob* cell_blob = this->GetCellBlob(BnInOp2Blob);
	const Blob* out_diff = BnInOp2Blob("out_diff");

	// reuse memory
	const Blob* f_gate_out = BnInOp2Blob("f_gate_out");
	Blob f_gate_diff_blob = BnInOp2Blob("f_gate_out");
	const Blob* i_gate_out = BnInOp2Blob("i_gate_out");
	Blob i_gate_diff_out = BnInOp2Blob("i_gate_out");
	const Blob* c_gate_out = BnInOp2Blob("c_gate_out");
	Blob c_diff_out = BnInOp2Blob("c_gate_out");
	const Blob* o_gate_out = BnInOp2Blob("o_gate_out");
	Blob o_diff_blob = BnInOp2Blob("o_gate_out");

	const Blob* f_out = BnInOp2Blob("f_out");
	Blob f_out_diff_blob = BnInOp2Blob("f_out");
	const Blob* i_out = BnInOp2Blob("i_out");
	Blob i_out_diff_out = BnInOp2Blob("i_out");
	const Blob* c_out = BnInOp2Blob("c_out");
	Blob c_out_diff_blob = BnInOp2Blob("c_out");
	const Blob* o_out = BnInOp2Blob("o_out");
	Blob o_out_diff_blob = BnInOp2Blob("o_out");
	
	
	//cell_out_diff = rec_out_diff * o_out * [1 - tanh(cell_out)*tanh(cell_out)] 
	//								+ cell_out_diff
	
	

	//i_out_diff = cell_out_diff * c_out	
	KernelUtil<device_type, T>::Mul(ctx.device_type, out_blob->shape().elem_cnt(),
																	cell_out_diff_blob->dptr<T>(), c_out_blob->dptr<T>(),
																	i_out_diff_blob->mut_dptr<T>());
	//	i_gate_out_diff = i_out_diff * (1 - i_out_diff)
	if (in_blob->col_id() == in_blob->max_col_id()) {
		(*last_colnum_activation_bw_func_)(
				ctx.device_ctx, out_blob->shape().elem_cnt(),
				)
	
	}
	//	h2h_i_weight_diff = i_gate_out_diff * hidden
	// 	i2h_i_weight_diff = i_gate_out_diff * in
	

	//f_out_diff = cell_out_diff * cell_in
	KernelUtil<device_type, T>::Mul(ctx.device_type, out_blob->shape().elem_cnt(),
																	cell_out_diff_blob->dptr<T>(), cell_in_blob->dptr<T>(),
																	f_out_diff_blob->mut_dptr<T>());
	//	f_gate_out_diff = f_out_diff * (1 - f_out_diff)
	//	h2h_f_weight_diff = f_gate_out_diff * hidden
	// 	i2h_f_weight_diff = f_gate_out_diff * in
	
	//c_out_diff = cell_out_diff * i_out
	KernelUtil<device_type, T>::Mul(ctx.device_type, out_blob->shape().elem_cnt(),
																	cell_out_diff_blob->dptr<T>(), i_out_blob->dptr<T>(),
																	c_out_diff_blob->mut_dptr<T>());
	//	c_gate_out_diff = c_out_diff * (1 - c_out_diff)
	//	h2h_c_weight_diff = c_gate_out_diff * hidden
	// 	i2h_c_weight_diff = c_gate_out_diff * in
	
	//o_out_diff = rec_out_diff * tanh(cell_out)
	KernelUtil<device_type, T>::TanH(ctx.device_type, out_blob->shape().elem_cnt(),
																	cell_out_blob->dptr<T>(), cell_out_blob->mut_dptr<T>());
	KernelUtil<device_type, T>::Mul(ctx.device_type, out_blob->shape().elem_cnt(),
																	rec_out_diff_blob->dptr<T>(), cell_out_blob->dptr<T>(),
																	o_out_diff_blob->mut_dptr<T>());

	//	o_gate_out_diff = o_out_diff * (1 - o_out_diff)
	//	h2h_o_weight_diff = o_gate_out_diff * hidden
	// 	i2h_o_weight_diff = o_gate_out_diff * in
	
}
	




template<DeviceType device_type, typename T>
void BasicLstmKernel<device_type, T>::InitPureModelTmpBlobs()

template<DeviceType device_type, typename T>
void BasicLstmKernel<device_type, T>::VitualInitModelBlobsWithDir()

template<DeviceType device_type, typename T>
void BasicLstmKernel<device_type, T>::VitualInitModelBlobsWithRandomSeed()


}


