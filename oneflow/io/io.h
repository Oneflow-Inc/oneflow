#ifndef _IO_IO_H_
#define _IO_IO_H_

#include <glog/logging.h>
#include <fstream>
#include <string>
#include <cstdint>
#include <vector>
#include <random>

namespace caffe {
// out binary stream
class BinaryOutputStream{
public:
  BinaryOutputStream();
  ~BinaryOutputStream();
  bool Open(const std::string file_name);
  bool WriteEmptyHeader(uint64_t* uint64_buf, uint64_t count);
  bool WriteRealHeader(uint64_t* uint64_buf, uint64_t count);
  bool WriteBlob(int32_t label, char* char_buf, uint64_t count);
  bool good() const { return good_; }
  bool Close();

private:
  // 200MB RAM for bin_buf_
  const uint64_t bin_buf_size_ = 1024 * 1024 * 200;

  std::ofstream stream_;
  std::string file_name_;

  char *bin_buf_{ nullptr };
  uint64_t buf_idx_{ 0 };

  bool good_{ false };

  bool seekp(uint64_t pos);

  BinaryOutputStream(const BinaryOutputStream& other) = delete;
  BinaryOutputStream& operator=(const BinaryOutputStream& other) = delete;
};

// in binary stream
class BinaryInputStream{
public:
  BinaryInputStream();
  ~BinaryInputStream();
  bool Open(const std::string file_name, bool do_shuffle = false);
  uint64_t get_total_num() const;
  uint64_t* get_offset() const;
  // get the next data and label, and idx_++ if success
  // return data size if success
  // return 0 if no data left
  uint64_t Next(std::vector<char>* data, int32_t *label);
  uint64_t Next(std::vector<char>* data, float *label);  // for float label
  bool good() const { return good_; }
  bool Close();

private:
  // 200MB RAM for bin_buf_
  const uint64_t bin_buf_size_ = 1024 * 1024 * 200;
  char *bin_buf_{ nullptr };
  uint64_t buf_offset_{ 0 };  // buffer offset of stream_
  uint64_t buf_data_size_{ 0 };  // real filled size

  std::ifstream stream_;
  std::string file_name_;

  uint64_t total_num_{ 0 };
  uint64_t idx_{ 0 };
  uint64_t *offset_{ nullptr };

  bool do_shuffle_{ false };
  std::vector<uint64_t> shuffle_order_;
  std::mt19937 rnd_;

  bool good_{ false };

  // read next chunk from disk
  // begin_in_buf, is the begin position of remaining data in bin_buf_
  void ReadNextChunk(uint64_t begin_in_buf);

  BinaryInputStream(const BinaryInputStream& other) = delete;
  BinaryInputStream& operator=(const BinaryInputStream& other) = delete;
};

}  // namespace caffe
#endif  // _IO_IO_H_
