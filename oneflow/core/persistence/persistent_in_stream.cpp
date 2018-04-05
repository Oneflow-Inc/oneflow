#include "oneflow/core/persistence/persistent_in_stream.h"
#include "oneflow/core/job/job_desc.h"

namespace oneflow {

PersistentInStream::~PersistentInStream() { delete[] buffer_; }

int32_t PersistentInStream::ReadLine(std::string* line) {
  if (IsEof()) { return -1; }
  line->clear();
  while (*cur_buf_begin_ != '\n') {
    if (cur_buf_begin_ == cur_buf_end_) {
      UpdateBuffer();
      if (cur_buf_begin_ == cur_buf_end_) {
        return 0;
      } else {
        continue;
      }
    }
    line->push_back(*cur_buf_begin_++);
  }
  ++cur_buf_begin_;
  return 0;
}

int32_t PersistentInStream::Read(char* s, size_t n) {
  if (IsEof()) { return -1; }
  while (n--) {
    if (cur_buf_begin_ == cur_buf_end_) { UpdateBuffer(); }
    CHECK_NE(cur_buf_begin_, cur_buf_end_);
    *s++ = *cur_buf_begin_++;
  }
  return 0;
}

PersistentInStream::PersistentInStream(fs::FileSystem* fs,
                                       const std::string& file_path,
                                       uint64_t offset) {
  fs->NewRandomAccessFile(file_path, &file_);
  file_size_ = fs->GetFileSize(file_path);
  cur_file_pos_ = offset;
  buffer_ =
      new char[Global<JobDesc>::Get()->persistence_buffer_byte_size() + 1];
  cur_buf_begin_ = buffer_;
  cur_buf_end_ = buffer_;
  *cur_buf_end_ = '\0';
}

bool PersistentInStream::IsEof() const {
  return cur_buf_begin_ == cur_buf_end_ && cur_file_pos_ == file_size_;
}

void PersistentInStream::UpdateBuffer() {
  CHECK_EQ(cur_buf_begin_, cur_buf_end_);
  uint64_t n = std::min(Global<JobDesc>::Get()->persistence_buffer_byte_size(),
                        file_size_ - cur_file_pos_);
  if (n == 0) { return; }
  file_->Read(cur_file_pos_, n, buffer_);
  cur_buf_begin_ = buffer_;
  cur_buf_end_ = buffer_ + n;
  *cur_buf_end_ = '\0';
  AddNForCurFilePos(n);
}

}  // namespace oneflow
