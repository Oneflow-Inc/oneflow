#include "oneflow/core/persistence/persistent_in_stream_without_local_copy.h"
#include "oneflow/core/job/job_desc.h"
#include <cstring>

namespace oneflow {

int32_t PersistentInStreamWithoutLocalCopy::ReadLine(std::string* l) {
  if (IsEof()) { return -1; }
  l->clear();
  while (*cur_buf_begin_ != '\n') {
    if (cur_buf_begin_ == cur_buf_end_) {
      UpdateBuffer();
      if (cur_buf_begin_ == cur_buf_end_) {
        return 0;
      } else {
        continue;
      }
    }
    l->push_back(*cur_buf_begin_++);
  }
  ++cur_buf_begin_;
  return 0;
}

int32_t PersistentInStreamWithoutLocalCopy::Read(char* s, size_t n) {
  if (IsEof()) { return -1; }
  while (n) {
    if (cur_buf_begin_ == cur_buf_end_) { UpdateBuffer(); }
    CHECK_LT(cur_buf_begin_, cur_buf_end_);
    int64_t copy_size =
        std::min(cur_buf_end_ - cur_buf_begin_, static_cast<int64_t>(n));
    std::memcpy(s, cur_buf_begin_, static_cast<size_t>(copy_size));
    s += copy_size;
    cur_buf_begin_ += copy_size;
    n -= copy_size;
  }
  return 0;
}

PersistentInStreamWithoutLocalCopy::PersistentInStreamWithoutLocalCopy(
    fs::FileSystem* fs, const std::string& file_path, uint64_t offset) {
  fs->NewRandomAccessFile(file_path, &file_);
  file_size_ = fs->GetFileSize(file_path);
  cur_file_pos_ = offset;
  buffer_.resize(Global<JobDesc>::Get()->persistence_buffer_byte_size() + 1);
  cur_buf_begin_ = buffer_.data();
  cur_buf_end_ = buffer_.data();
  *cur_buf_end_ = '\0';
}

bool PersistentInStreamWithoutLocalCopy::IsEof() const {
  return cur_buf_begin_ == cur_buf_end_ && cur_file_pos_ == file_size_;
}

void PersistentInStreamWithoutLocalCopy::UpdateBuffer() {
  CHECK_EQ(cur_buf_begin_, cur_buf_end_);
  uint64_t n = std::min(buffer_.size() - 1, file_size_ - cur_file_pos_);
  if (n == 0) { return; }
  file_->Read(cur_file_pos_, n, buffer_.data());
  cur_buf_begin_ = buffer_.data();
  cur_buf_end_ = buffer_.data() + n;
  *cur_buf_end_ = '\0';
  AddNForCurFilePos(n);
}

}  // namespace oneflow
