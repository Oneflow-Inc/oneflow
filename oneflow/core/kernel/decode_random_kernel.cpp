#include "oneflow/core/kernel/decode_random_kernel.h"

namespace oneflow {

namespace {

void RandomFillBlob(DeviceCtx* ctx, const InitializerConf& initializer_conf, uint32_t random_seed,
                    Blob* blob) {
  static const HashMap<int, void (*)(DeviceCtx * ctx, const InitializerConf& initializer_conf,
                                     uint32_t random_seed, Blob* blob)>
      fill_funcs = {
#define RANDOM_FILL_ENTRY(type_cpp, type_proto) \
  {type_proto, &KernelUtil<DeviceType::kCPU, type_cpp>::InitializeWithConf},
          OF_PP_FOR_EACH_TUPLE(RANDOM_FILL_ENTRY, ARITHMETIC_DATA_TYPE_SEQ)};
  fill_funcs.at(blob->data_type())(ctx, initializer_conf, random_seed, blob);
}

}  // namespace

void DecodeRandomKernel::VirtualKernelInit(const ParallelContext*) {
  gen_.reset(new std::mt19937(kernel_conf().decode_random_conf().random_seed()));
  dis_.reset(new std::uniform_int_distribution<uint32_t>());
}

uint32_t DecodeRandomKernel::GenNextRandomSeed() const { return (*dis_)(*gen_); }

void DecodeRandomKernel::Forward(const KernelCtx& ctx,
                                 std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const DecodeRandomOpConf& conf = op_conf().decode_random_conf();
  RandomFillBlob(ctx.device_ctx, conf.initializer(), GenNextRandomSeed(),
                 BnInOp2Blob(op_attribute().output_bns(0)));
}

REGISTER_KERNEL(OperatorConf::kDecodeRandomConf, DecodeRandomKernel);

}  // namespace oneflow
