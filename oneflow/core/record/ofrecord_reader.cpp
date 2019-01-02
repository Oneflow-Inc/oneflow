#include "oneflow/core/record/ofrecord_reader.h"

namespace oneflow {

constexpr int64_t MAX_CHUNK_SIZE = 64 * 1024 * 1024;  // 64M

namespace {

bool ReadChunk(PersistentInStream* is, OFRecordChunk* chunk) {
  if (is->Read(reinterpret_cast<char*>(&chunk->size), sizeof(int64_t)) == 0) {
    CHECK_GE(chunk->size, 0);
    CHECK_LE(chunk->size, MAX_CHUNK_SIZE);
    chunk->data.reset(new char[chunk->size]);
    CHECK_EQ(is->Read(chunk->data.get(), chunk->size), 0);
    return true;
  }
  return false;
}

}  // namespace

NaiveOFRecordReader::NaiveOFRecordReader(PersistentInStream* in, size_t num_max_read)
    : in_stream_(in), num_read_(0), num_max_read_(num_max_read) {}

size_t NaiveOFRecordReader::Read(size_t n, OFRecord* allocated_records) {
  OFRecordChunk chunk;
  const size_t can_read = std::min(n, num_max_read_ - num_read_);
  FOR_RANGE(size_t, i, 0, can_read) {
    if (ReadChunk(in_stream_, &chunk)) {
      CHECK(allocated_records[i].ParseFromArray(chunk.data.get(), chunk.size));
      ++num_read_;
    } else {
      return i;
    }
  }
  return can_read;
}

RandomShuffleOFRecordReader::RandomShuffleOFRecordReader(PersistentInStream* in, size_t buffer_size,
                                                         size_t num_max_read, int32_t random_seed)
    : in_stream_(in),
      buffer_size_(buffer_size),
      num_max_read_(num_max_read),
      random_gen_(random_seed),
      is_eof_(false) {
  CHECK_GT(buffer_size, 0);
  buffered_chunks_.reserve(buffer_size);
}

void RandomShuffleOFRecordReader::FillBuffer() {
  for (; num_read_ < num_max_read_ && buffered_chunks_.size() < buffer_size_; ++num_read_) {
    OFRecordChunk chunk;
    if (ReadChunk(in_stream_, &chunk)) {
      buffered_chunks_.emplace_back(std::move(chunk));
    } else {
      is_eof_ = true;
      break;
    }
  }
  if (num_read_ == num_max_read_) { is_eof_ = true; }
}

size_t RandomShuffleOFRecordReader::Read(size_t n, OFRecord* allocated_records) {
  size_t cur_read = 0;
  while (cur_read < n) {
    if (!is_eof_) { FillBuffer(); }
    if (buffered_chunks_.empty()) { break; }
    const size_t pos =
        std::uniform_int_distribution<size_t>(0, buffered_chunks_.size() - 1)(random_gen_);
    if (pos != buffered_chunks_.size() - 1) {
      std::swap(buffered_chunks_[pos], buffered_chunks_.back());
    }
    CHECK(allocated_records[cur_read].ParseFromArray(buffered_chunks_.back().data.get(),
                                                     buffered_chunks_.back().size));
    buffered_chunks_.pop_back();
    ++cur_read;
  }
  return cur_read;
}

}  // namespace oneflow
