#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/record/onerec_reader.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/record/onerec_decoder.h"
#include "oneflow/core/common/protobuf.h"

namespace oneflow {

namespace {

class RoundRobinOneRecDecoder final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RoundRobinOneRecDecoder);
  RoundRobinOneRecDecoder(const std::vector<std::string>& files,
                          const std::vector<DecodeOneRecFieldConf>& fields, int32_t batch_size,
                          int32_t buffer_size, int32_t num_threads) {
    CHECK_GT(files.size(), 0);
    CHECK_GT(batch_size, 0);
    CHECK_GT(num_threads, 0);
    CHECK_EQ(files.size() % num_threads, 0);
    BalancedSplitter bs(files.size(), num_threads);
    for (int64_t i = 0; i < num_threads; ++i) {
      const std::vector<std::string> data_paths(
          {files.cbegin() + bs.At(i).begin(), files.cbegin() + bs.At(i).end()});
      std::unique_ptr<PersistentInStream> in_stream(
          new PersistentInStream(DataFS(), data_paths, true, false));
      std::unique_ptr<BufferedBatchedOneRecReader> reader(new BufferedBatchedOneRecReader(
          std::move(in_stream), GetMaxVal<int64_t>(), batch_size, buffer_size));
      decoder_.emplace_back(new OneRecDecoder(std::move(reader), batch_size, fields));
    }
  }
  ~RoundRobinOneRecDecoder() = default;

  bool GetBatch(const std::vector<Blob*>& blobs) {
    OneRecDecoder* decoder = decoder_.at(counter_ % decoder_.size()).get();
    counter_ += 1;
    return decoder->GetBatch(blobs);
  }

 private:
  int64_t counter_ = 0;
  std::vector<std::unique_ptr<OneRecDecoder>> decoder_;
};

}  // namespace

class DecodeOneRecKernel final : public KernelIf<DeviceType::kCPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DecodeOneRecKernel);
  DecodeOneRecKernel() = default;
  ~DecodeOneRecKernel() override = default;

 private:
  void VirtualKernelInit() override;
  void Forward(const KernelCtx& ctx,
               std::function<Blob*(const std::string&)> BnInOp2Blob) const override;

  std::unique_ptr<RoundRobinOneRecDecoder> decoder_;
};

void DecodeOneRecKernel::VirtualKernelInit() {
  const DecodeOneRecKernelConf& conf = this->kernel_conf().decode_onerec_conf();
  decoder_.reset(new RoundRobinOneRecDecoder(
      PbRpf2StdVec(conf.file()), PbRpf2StdVec(this->op_conf().decode_onerec_conf().field()),
      conf.device_batch_size(), conf.buffer_size(), conf.num_threads()));
}

void DecodeOneRecKernel::Forward(const KernelCtx& ctx,
                                 std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  std::vector<Blob*> blobs;
  blobs.reserve(this->op_attribute().output_bns().size());
  for (const std::string& bn : this->op_attribute().output_bns()) {
    blobs.push_back(BnInOp2Blob(bn));
  }
  CHECK(decoder_->GetBatch(blobs));
}

REGISTER_KERNEL(OperatorConf::kDecodeOnerecConf, DecodeOneRecKernel);

}  // namespace oneflow
