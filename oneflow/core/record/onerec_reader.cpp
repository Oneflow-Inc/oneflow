#include "oneflow/core/record/onerec_reader.h"
#include "oneflow/core/common/blocking_counter.h"
#include <onerec/example_generated.h>

#define XXH_NAMESPACE LZ4_
#include <xxhash.h>

namespace oneflow {

namespace {

struct OneRecFrame {
  std::shared_ptr<char> payload;
  int32_t payload_size = 0;
};

constexpr int64_t kMaxPayloadSize = std::numeric_limits<int32_t>::max();
constexpr int64_t kMagicNumber = 0x24434552454E4F5E;  // '^ONEREC$', little endian
constexpr int32_t kReservedNumber = 0;
constexpr int32_t kPayloadAlignmentSize = 8;
constexpr int32_t kMagicFieldSize = 8;
constexpr int32_t kReservedFieldSize = 4;
constexpr int32_t kPayloadSizeFieldSize = 4;
constexpr int32_t kDigestFieldSize = 8;
constexpr int32_t kHeaderSizeWithoutDigest =
    kMagicFieldSize + kReservedFieldSize + kPayloadSizeFieldSize;
constexpr int32_t kHeaderSize = kHeaderSizeWithoutDigest + kDigestFieldSize;

inline XXH64_hash_t ByteSwap(XXH64_hash_t x) {
  return ((x & 0xff00000000000000ull) >> 56u) | ((x & 0x00ff000000000000ull) >> 40u)
         | ((x & 0x0000ff0000000000ull) >> 24u) | ((x & 0x000000ff00000000ull) >> 8u)
         | ((x & 0x00000000ff000000ull) << 8u) | ((x & 0x0000000000ff0000ull) << 24u)
         | ((x & 0x000000000000ff00ull) << 40u) | ((x & 0x00000000000000ffull) << 56u);
}

struct OneRecFrameHeader {
  int64_t magic;
  int32_t reserved;
  int32_t payload_size;
  XXH64_hash_t digest;
};

union OneRecFrameHeaderView {
  char raw[kHeaderSize];
  OneRecFrameHeader header;
};

union OneRecFrameFooterView {
  char raw[kDigestFieldSize];
  XXH64_hash_t digest;
};

bool ReadOneRecFrame(PersistentInStream* is, OneRecFrame* frame) {
  static_assert(sizeof(OneRecFrameHeader) == kHeaderSize, "");
  OneRecFrameHeaderView header_view{};
  static_assert(sizeof(header_view.header) == kHeaderSize, "");
  if (is->ReadFully(header_view.raw, kHeaderSize) != 0) { return false; }
  CHECK_EQ(header_view.header.magic, kMagicNumber);
  CHECK_EQ(header_view.header.reserved, kReservedNumber);
  const int32_t payload_size = header_view.header.payload_size;
  CHECK_GE(payload_size, 0);
  CHECK_LE(payload_size, kMaxPayloadSize);
  XXH64_state_t* const state = LZ4_XXH64_createState();
  CHECK_NOTNULL(state);
  XXH64_hash_t const seed = 0;
  CHECK_NE(LZ4_XXH64_reset(state, seed), XXH_ERROR);
  CHECK_NE(XXH64_update(state, header_view.raw, kHeaderSizeWithoutDigest), XXH_ERROR);
  CHECK_EQ(ByteSwap(header_view.header.digest), LZ4_XXH64_digest(state));
  const int32_t padded_size = RoundUp(payload_size, kPayloadAlignmentSize);
  const int32_t body_size = padded_size + kDigestFieldSize;
  char* body = reinterpret_cast<char*>(malloc(body_size));
  CHECK_EQ(is->ReadFully(body, body_size), 0);
  static_assert(sizeof(OneRecFrameFooterView) == kDigestFieldSize, "");
  OneRecFrameFooterView footer_view{};
  std::memcpy(&footer_view, body + padded_size, sizeof(OneRecFrameFooterView));
  CHECK_NE(XXH64_reset(state, seed), XXH_ERROR);
  CHECK_NE(LZ4_XXH64_update(state, body, payload_size), XXH_ERROR);
  CHECK_EQ(ByteSwap(footer_view.digest), LZ4_XXH64_digest(state));
  CHECK_NE(LZ4_XXH64_freeState(state), XXH_ERROR);
  frame->payload_size = payload_size;
  frame->payload.reset(body);
  return true;
}

}  // namespace

BufferedBatchedOneRecReader::BufferedBatchedOneRecReader(PersistentInStream* in,
                                                         size_t num_max_read, size_t batch_size,
                                                         size_t buffer_size)
    : in_stream_(in),
      num_read_(0),
      num_max_read_(num_max_read),
      batch_size_(batch_size),
      buffer_size_(buffer_size),
      buffer_(buffer_size) {
  reader_thread_ = std::thread([&]() {
    while (num_read_ < num_max_read_) {
      size_t cur_batch_size = std::min(batch_size_, num_max_read_ - num_read_);
      std::vector<OneRecExampleWrapper> cur_batch;
      cur_batch.reserve(cur_batch_size);
      OneRecFrame frame;
      for (size_t i = 0; i < cur_batch_size; ++i) {
        if (ReadOneRecFrame(in_stream_, &frame)) {
          cur_batch.emplace_back(frame.payload, frame.payload_size);
        }
      }
      const size_t read_batch_size = cur_batch.size();
      if (read_batch_size > 0) {
        const BufferStatus status = buffer_.Send(cur_batch);
        if (status == BufferStatus::kBufferStatusErrorClosed) {
          break;
        } else {
          CHECK(status == BufferStatus::kBufferStatusSuccess);
        }
      }
      num_read_ += read_batch_size;
      if (read_batch_size < batch_size_) {
        buffer_.Close();
        break;
      }
    }
  });
}

BufferedBatchedOneRecReader::~BufferedBatchedOneRecReader() {
  buffer_.Close();
  reader_thread_.join();
}

void BufferedBatchedOneRecReader::Read(std::vector<OneRecExampleWrapper>* batch) {
  CHECK(batch->empty());
  BufferStatus status = buffer_.Receive(batch);
  if (status != BufferStatus::kBufferStatusSuccess) {
    CHECK(status == BufferStatus::kBufferStatusErrorClosed);
  }
  CHECK_EQ(status, BufferStatus::kBufferStatusSuccess);
}

}  // namespace oneflow
