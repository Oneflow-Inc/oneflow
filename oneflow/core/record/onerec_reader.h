#ifndef ONEFLOW_CORE_RECORD_ONEREC_READER_H_
#define ONEFLOW_CORE_RECORD_ONEREC_READER_H_

#include <onerec/example_generated.h>
#include "oneflow/core/common/util.h"
#include "oneflow/core/persistence/persistent_in_stream.h"
#include "oneflow/core/common/buffer.h"

namespace oneflow {

class OneRecExampleWrapper {
 public:
  OneRecExampleWrapper(std::shared_ptr<char> data, int32_t size)
      : size_(size), data_(std::move(data)) {
    const auto buffer = reinterpret_cast<const uint8_t*>(data_.get());
    flatbuffers::Verifier verifier(buffer, static_cast<size_t>(size_));
    CHECK(onerec::example::VerifyExampleBuffer(verifier));
    example_ = onerec::example::GetExample(buffer);
  }
  ~OneRecExampleWrapper() = default;

  const onerec::example::Example* GetExample() { return example_; }

 private:
  int32_t size_;
  std::shared_ptr<char> data_;
  const onerec::example::Example* example_;
};

class BufferedBatchedOneRecReader final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BufferedBatchedOneRecReader);
  BufferedBatchedOneRecReader(PersistentInStream* in, size_t batch_size, size_t buffer_size)
      : BufferedBatchedOneRecReader(in, GetMaxVal<int64_t>(), batch_size, buffer_size) {}
  BufferedBatchedOneRecReader(PersistentInStream* in, size_t num_max_read, size_t batch_size,
                              size_t buffer_size);
  ~BufferedBatchedOneRecReader();

  void Read(std::vector<OneRecExampleWrapper>* batch);

 private:
  PersistentInStream* in_stream_;
  size_t num_read_;
  const size_t num_max_read_;
  const size_t batch_size_;
  const size_t buffer_size_;
  Buffer<std::vector<OneRecExampleWrapper>> buffer_;
  std::thread reader_thread_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_RECORD_ONEREC_READER_H_
