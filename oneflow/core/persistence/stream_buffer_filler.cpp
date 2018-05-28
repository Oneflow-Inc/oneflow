#include <vector>
#include "oneflow/core/persistence/stream_buffer_filler.h"
#include "oneflow/core/persistence/binary_in_stream_without_local_copy.h"
#include "oneflow/core/persistence/binary_in_stream_with_local_copy.h"

namespace oneflow {

StreamBufferFiller::StreamBufferFiller(fs::FileSystem* fs, const std::string& file_path_prefix,
                                       int32_t min_id, int32_t max_id, bool cyclic,
                                       bool with_local_copy)
    : cur_stream_id_(0), cyclic_(cyclic), with_local_copy_(with_local_copy) {
  CHECK_LE(min_id, max_id);
  LOG(INFO) << "New StreamBufferFiller " << file_path_prefix << "-[" << min_id << "," << max_id
            << "]";

  stream_num_ = max_id - min_id + 1;
  whole_file_size_ = 0;
  FOR_RANGE(int32_t, part_id, min_id, max_id + 1) {
    std::string file_path = file_path_prefix + std::to_string(part_id);
    AddStream(fs, file_path, 0);
  }
  whole_file_pos_ = 0;
}

StreamBufferFiller::StreamBufferFiller(fs::FileSystem* fs, const std::string& file_path,
                                       uint64_t offset, bool cyclic, bool with_local_copy)
    : cur_stream_id_(0), cyclic_(cyclic), with_local_copy_(with_local_copy) {
  LOG(INFO) << "New StreamBufferFiller " << file_path;
  stream_num_ = 1;
  whole_file_size_ = 0;
  AddStream(fs, file_path, offset);
  whole_file_pos_ = offset;
}

void StreamBufferFiller::AddStream(fs::FileSystem* fs, const std::string& file_path,
                                   uint64_t offset) {
  if (with_local_copy_) {
    streams_.emplace_back(new BinaryInStreamWithLocalCopy(fs, file_path));
  } else {
    streams_.emplace_back(new BinaryInStreamWithoutLocalCopy(fs, file_path, offset));
  }
  uint64_t cur_file_size = fs->GetFileSize(file_path);
  whole_file_size_ += cur_file_size;
}

bool StreamBufferFiller::IsEof() const { return whole_file_pos_ == whole_file_size_; }

uint64_t StreamBufferFiller::UpdateBuffer(std::vector<char>* buffer) {
  if (cur_stream_id_ == stream_num_) return 0;
  uint64_t n = std::min(buffer->size() - 1, streams_[cur_stream_id_]->file_size()
                                                - streams_[cur_stream_id_]->cur_file_pos());
  if (n == 0) { return 0; }
  streams_[cur_stream_id_]->Read(buffer->data(), n);

  if (cyclic_) {
    AddNForCurFilePosNonCyclic(n);
  } else {
    AddNForCurFilePosCyclic(n);
  }
  return n;
}

void StreamBufferFiller::AddNForCurFilePosNonCyclic(uint64_t n) {
  whole_file_pos_ += n;
  if (streams_[cur_stream_id_]->IsEof()) { ++cur_stream_id_; }
}

void StreamBufferFiller::AddNForCurFilePosCyclic(uint64_t n) {
  whole_file_pos_ = (whole_file_pos_ + n) % whole_file_size_;
  if (streams_[cur_stream_id_]->IsEof()) {
    streams_[cur_stream_id_]->set_cur_file_pos(0);
    ++cur_stream_id_;
    if (cur_stream_id_ == stream_num_) {
      CHECK_EQ(whole_file_pos_, whole_file_size_);
      cur_stream_id_ = 0;
    }
  }
}
}  // namespace oneflow
