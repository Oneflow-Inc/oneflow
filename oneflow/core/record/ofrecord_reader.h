#ifndef ONEFLOW_CORE_RECORD_OFRECORD_READER_H_
#define ONEFLOW_CORE_RECORD_OFRECORD_READER_H_

#include "oneflow/core/record/record.pb.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/persistence/persistent_in_stream.h"

namespace oneflow {

struct OFRecordChunk {
  int64_t size = 0;
  std::unique_ptr<char[]> data;
};

class OFRecordReader {
 public:
  OF_DISALLOW_COPY_AND_MOVE(OFRecordReader);
  OFRecordReader() = default;
  virtual ~OFRecordReader() = default;

  virtual size_t Read(size_t n, OFRecord* allocated_records) = 0;
};

class NaiveOFRecordReader final : public OFRecordReader {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NaiveOFRecordReader);
  explicit NaiveOFRecordReader(PersistentInStream* in)
      : NaiveOFRecordReader(in, GetMaxVal<size_t>()) {}
  NaiveOFRecordReader(PersistentInStream* in, size_t num_max_read);
  ~NaiveOFRecordReader() override = default;

 private:
  size_t Read(size_t n, OFRecord* allocated_records) override;

  PersistentInStream* in_stream_;
  size_t num_read_;
  const size_t num_max_read_;
};

class RandomShuffleOFRecordReader final : public OFRecordReader {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RandomShuffleOFRecordReader);
  RandomShuffleOFRecordReader(PersistentInStream* in, size_t buffer_size, size_t num_max_read,
                              int32_t random_seed);
  RandomShuffleOFRecordReader(PersistentInStream* in, size_t buffer_size, size_t num_max_read)
      : RandomShuffleOFRecordReader(in, buffer_size, num_max_read, std::random_device()()) {}
  RandomShuffleOFRecordReader(PersistentInStream* in, size_t buffer_size)
      : RandomShuffleOFRecordReader(in, buffer_size, GetMaxVal<size_t>()) {}
  ~RandomShuffleOFRecordReader() override = default;

 private:
  size_t Read(size_t n, OFRecord* allocated_records) override;
  void FillBuffer();

  PersistentInStream* in_stream_;
  const size_t buffer_size_;
  const size_t num_max_read_;
  std::mt19937 random_gen_;
  size_t num_read_;
  std::vector<OFRecordChunk> buffered_chunks_;
  bool is_eof_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_RECORD_OFRECORD_READER_H_
