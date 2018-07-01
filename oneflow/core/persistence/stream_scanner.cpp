#include "oneflow/core/persistence/stream_scanner.h"
#include "oneflow/core/persistence/binary_in_stream_without_local_copy.h"
#include "oneflow/core/persistence/binary_in_stream_with_local_copy.h"

namespace oneflow {

StreamScanner::StreamScanner(fs::FileSystem* fs, const std::vector<std::string>& file_paths,
                             uint64_t offset, bool with_local_copy)
    : whole_file_offset_(offset), with_local_copy_(with_local_copy) {
  if (with_local_copy_) { CHECK_EQ(whole_file_offset_, 0); }
  stream_num_ = file_paths.size();
  whole_file_size_ = 0;
  int64_t idx = 0;
  for (auto& file_path : file_paths) {
    AddStream(fs, file_path, idx);
    ++idx;
  }
  CHECK_LE(whole_file_offset_, whole_file_size_);
  whole_file_pos_ = whole_file_offset_;
}

void StreamScanner::AddStream(fs::FileSystem* fs, const std::string& file_path, int64_t idx) {
  uint64_t cur_file_size = fs->GetFileSize(file_path);
  uint64_t offset;
  if (whole_file_offset_ < whole_file_size_) { offset = 0; }
  if (whole_file_offset_ > whole_file_size_) {
    offset = cur_file_size;
  } else if (whole_file_size_ <= whole_file_offset_
             && whole_file_offset_ < whole_file_size_ + cur_file_size) {
    offset = whole_file_offset_ - whole_file_size_;
    cur_stream_id_ = idx;
  } else {
    // FIXME
    /*
    if (cyclic_) {
      offset = 0;
    } else {
      offset = cur_file_size;
    }
    */
  }

  if (with_local_copy_) {
    streams_.emplace_back(new BinaryInStreamWithLocalCopy(fs, file_path));
  } else {
    streams_.emplace_back(new BinaryInStreamWithoutLocalCopy(fs, file_path, offset));
  }
  whole_file_size_ += cur_file_size;
}

bool StreamScanner::IsEof() const { return whole_file_pos_ == whole_file_size_; }

uint64_t StreamScanner::UpdateBuffer(std::vector<char>* buffer) {
  if (cur_stream_id_ == stream_num_) return 0;
  uint64_t n = std::min(buffer->size() - 1, streams_[cur_stream_id_]->file_size()
                                                - streams_[cur_stream_id_]->cur_file_pos());
  if (n == 0) { return 0; }
  streams_[cur_stream_id_]->Read(buffer->data(), n);
  AddNForCurFilePos(n);
  return n;
}

void AcyclicStreamScanner::AddNForCurFilePos(uint64_t n) {
  whole_file_pos_ += n;
  if (streams_[cur_stream_id_]->IsEof()) { ++cur_stream_id_; }
}

void CyclicStreamScanner::AddNForCurFilePos(uint64_t n) {
  whole_file_pos_ = (whole_file_pos_ + n) % whole_file_size_;
  if (streams_[cur_stream_id_]->IsEof()) {
    streams_[cur_stream_id_]->set_cur_file_pos(0);
    ++cur_stream_id_;
    if (cur_stream_id_ == stream_num_) {
      CHECK_EQ(whole_file_pos_, 0);
      cur_stream_id_ = 0;
    }
  }
}

}  // namespace oneflow
