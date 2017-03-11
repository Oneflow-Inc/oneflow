#include "io/io.h"
#include <time.h>
#include <string>
#include <vector>
#include <algorithm>

namespace caffe {
// BinaryOutputStream
BinaryOutputStream::BinaryOutputStream() {}
BinaryOutputStream::~BinaryOutputStream() {
  if (bin_buf_) {
    delete[]bin_buf_;
    bin_buf_ = nullptr;
  }
}

bool BinaryOutputStream::Open(const std::string file_name) {
  file_name_ = file_name;
  stream_.open(file_name_, std::ios::out | std::ios::binary);
  if (stream_.good()) {
    bin_buf_ = new char[bin_buf_size_];
    good_ = true;
    return good_;
  }
  return false;
}
bool BinaryOutputStream::seekp(uint64_t pos) {
  CHECK(good_) << "File should be opened.";
  stream_.seekp(pos);
  return true;
}

bool BinaryOutputStream::WriteEmptyHeader(uint64_t* uint64_buf,
  uint64_t count) {
  CHECK(good_) << "File should be opened.";
  stream_.write(reinterpret_cast<char*>(&count), sizeof(uint64_t));
  stream_.write(reinterpret_cast<char*>(uint64_buf),
    sizeof(uint64_t)* (count + 1));
  return true;
}
bool BinaryOutputStream::WriteRealHeader(uint64_t* uint64_buf,
  uint64_t count) {
  CHECK(good_) << "File should be opened.";
  // clear off the block_buf_, if any content not dumped to disk
  if (buf_idx_ != 0) {
    stream_.write(reinterpret_cast<char*> (bin_buf_),
      sizeof(char)* buf_idx_);
    buf_idx_ = 0;
  }
  seekp(0);
  WriteEmptyHeader(uint64_buf, count);
  return true;
}

bool BinaryOutputStream::WriteBlob(int32_t label, char* char_buf,
  uint64_t count) {
  CHECK(good_) << "File should be opened.";
  if (buf_idx_ + count > bin_buf_size_) {
    stream_.write(reinterpret_cast<char*>(bin_buf_),
      sizeof(char)* buf_idx_);
    buf_idx_ = 0;
  }
  // add label
  memcpy(bin_buf_ + buf_idx_, &label, sizeof(int32_t));
  buf_idx_ += sizeof(int32_t);
  // add data
  memcpy(bin_buf_ + buf_idx_, char_buf, count * sizeof(char));
  buf_idx_ += count;
  return true;
}
bool BinaryOutputStream::Close() {
  CHECK(good_) << "File should be opened.";
  good_ = false;
  stream_.close();
  if (bin_buf_) {
    delete[]bin_buf_;
    bin_buf_ = nullptr;
  }
  return true;
}

// BinaryInputStream
BinaryInputStream::BinaryInputStream() {}
BinaryInputStream::~BinaryInputStream() {
  if (bin_buf_) {
    delete[]bin_buf_;
    bin_buf_ = nullptr;
  }
  if (offset_) {
    delete[]offset_;
    offset_ = nullptr;
  }
}
bool BinaryInputStream::Open(const std::string file_name,
  bool do_shuffle) {
  CHECK(!good_) << "File has been opened.";
  file_name_ = file_name;
  stream_.open(file_name_, std::ios::in | std::ios::binary);
  if (stream_.good()) {
    bin_buf_ = new char[bin_buf_size_];
    // read total_num_
    stream_.read(reinterpret_cast<char*>(&total_num_), sizeof(uint64_t));
    // read offset_
    offset_ = new uint64_t[total_num_ + 1];
    stream_.read(reinterpret_cast<char*>(offset_),
      sizeof(uint64_t)* (total_num_ + 1));

    // init random engine
    if (do_shuffle) {
      do_shuffle_ = do_shuffle;
      rnd_.seed(time(0));
    }
    // read first buf
    ReadNextChunk(0);
    good_ = true;
    return true;
  }
  return false;
}
bool BinaryInputStream::Close() {
  CHECK(good_) << "File should be opened.";
  stream_.close();
  if (bin_buf_) {
    delete[]bin_buf_;
    bin_buf_ = nullptr;
  }
  if (offset_) {
    delete[]offset_;
    offset_ = nullptr;
  }
  good_ = false;
  return true;
}
uint64_t BinaryInputStream::get_total_num() const {
  CHECK(good_) << "File should be opened.";
  return total_num_;
}
uint64_t* BinaryInputStream::get_offset() const {
  CHECK(good_) << "File should be opened.";
  return offset_;
}
uint64_t BinaryInputStream::Next(std::vector<char>* data, int32_t *label) {
  CHECK(good_) << "File should be opened.";
  if (idx_ < total_num_) {
    uint64_t real_idx = idx_;
    if (do_shuffle_) {
      if (!shuffle_order_.empty()) {
        // if shuffled, we should get the real id of data by idx_
        real_idx = shuffle_order_.back();
        shuffle_order_.pop_back();
      }
    }
    uint64_t begin_in_buf = offset_[real_idx] - buf_offset_;
    uint64_t end_in_buf = offset_[real_idx + 1] - buf_offset_;
    uint64_t datasize = end_in_buf - begin_in_buf - sizeof(int32_t);
    if (end_in_buf <= buf_data_size_) {
      // all in buf, so we can copy it directly from buf
      data->resize(datasize);
      memcpy(label, bin_buf_ + begin_in_buf, sizeof(int32_t));
      memcpy(&(*data)[0], bin_buf_ + begin_in_buf + sizeof(int32_t), datasize);
      idx_++;
      return datasize;
    } else {
      // not all in buf, so we should get the buf filled
      CHECK_EQ(idx_, real_idx) << "error";
      CHECK(buf_data_size_ == bin_buf_size_)
        << "get next error: file ends error.";
      ReadNextChunk(begin_in_buf);  // get next chunk
      return Next(data, label);  // then try get next again
    }
  }
  return 0;
}
uint64_t BinaryInputStream::Next(std::vector<char>* data, float *label) {
  int32_t *tmp_label = new int32_t;
  uint64_t tmp_size = Next(data, tmp_label);
  *label = *tmp_label;
  delete tmp_label;
  return tmp_size;
}

void BinaryInputStream::ReadNextChunk(uint64_t begin_in_buf) {
  uint64_t size_left = 0;
  if (begin_in_buf < buf_data_size_) {
    // if there are remaining data not used in buf
    size_left = buf_data_size_ - begin_in_buf;
    memmove(bin_buf_, bin_buf_ + begin_in_buf, size_left);
  }

  // read from disk
  stream_.read(bin_buf_ + size_left, bin_buf_size_ - size_left);
  buf_offset_ = buf_offset_ + buf_data_size_ - size_left;  // new offset
  buf_data_size_ = size_left + stream_.gcount();  // new datasize

  if (do_shuffle_) {
    // shuffle the id in buf
    uint64_t i = idx_;
    uint64_t tmpsize = buf_data_size_ + buf_offset_;
    while (offset_[i + 1] < tmpsize) {
      shuffle_order_.push_back(i);
      i++;
    }
    shuffle(shuffle_order_.begin(), shuffle_order_.end(), rnd_);
  }
}

}  // namespace caffe
