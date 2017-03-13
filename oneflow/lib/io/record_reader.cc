/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/lib/io/record_reader.h"

#include <limits.h>

#include "tensorflow/core/lib/core/coding.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/hash/crc32c.h"
#include "tensorflow/core/platform/env.h"

namespace tensorflow {
namespace io {

RecordReader::RecordReader(RandomAccessFile* file,
                           const RecordReaderOptions& options)
    : src_(file), options_(options) {
  if (options.compression_type == RecordReaderOptions::ZLIB_COMPRESSION) {
// We don't have zlib available on all embedded platforms, so fail.
#if defined(IS_SLIM_BUILD)
    LOG(FATAL) << "Zlib compression is unsupported on mobile platforms.";
#else   // IS_SLIM_BUILD
    zlib_input_buffer_.reset(new ZlibInputBuffer(
        src_, options.zlib_options.input_buffer_size,
        options.zlib_options.output_buffer_size, options.zlib_options));
#endif  // IS_SLIM_BUILD
  } else if (options.compression_type == RecordReaderOptions::NONE) {
    // Nothing to do.
  } else {
    LOG(FATAL) << "Unspecified compression type :" << options.compression_type;
  }
}

RecordReader::~RecordReader() {}

// Read n+4 bytes from file, verify that checksum of first n bytes is
// stored in the last 4 bytes and store the first n bytes in *result.
// May use *storage as backing store.
Status RecordReader::ReadChecksummed(uint64 offset, size_t n,
                                     StringPiece* result, string* storage) {
  if (n >= SIZE_MAX - sizeof(uint32)) {
    return errors::DataLoss("record size too large");
  }

  const size_t expected = n + sizeof(uint32);
  storage->resize(expected);

#if !defined(IS_SLIM_BUILD)
  if (zlib_input_buffer_) {
    // If we have a zlib compressed buffer, we assume that the
    // file is being read sequentially, and we use the underlying
    // implementation to read the data.
    //
    // No checks are done to validate that the file is being read
    // sequentially.  At some point the zlib input buffer may support
    // seeking, possibly inefficiently.
    TF_RETURN_IF_ERROR(zlib_input_buffer_->ReadNBytes(expected, storage));

    if (storage->size() != expected) {
      if (storage->size() == 0) {
        return errors::OutOfRange("eof");
      } else {
        return errors::DataLoss("truncated record at ", offset);
      }
    }

    uint32 masked_crc = core::DecodeFixed32(storage->data() + n);
    if (crc32c::Unmask(masked_crc) != crc32c::Value(storage->data(), n)) {
      return errors::DataLoss("corrupted record at ", offset);
    }
    *result = StringPiece(storage->data(), n);
  } else {
#endif  // IS_SLIM_BUILD
    // This version supports reading from arbitrary offsets
    // since we are accessing the random access file directly.
    StringPiece data;
    TF_RETURN_IF_ERROR(src_->Read(offset, expected, &data, &(*storage)[0]));
    if (data.size() != expected) {
      if (data.size() == 0) {
        return errors::OutOfRange("eof");
      } else {
        return errors::DataLoss("truncated record at ", offset);
      }
    }
    uint32 masked_crc = core::DecodeFixed32(data.data() + n);
    if (crc32c::Unmask(masked_crc) != crc32c::Value(data.data(), n)) {
      return errors::DataLoss("corrupted record at ", offset);
    }
    *result = StringPiece(data.data(), n);
#if !defined(IS_SLIM_BUILD)
  }
#endif  // IS_SLIM_BUILD

  return Status::OK();
}

Status RecordReader::ReadRecord(uint64* offset, string* record) {
  static const size_t kHeaderSize = sizeof(uint64) + sizeof(uint32);
  static const size_t kFooterSize = sizeof(uint32);

  // Read header data.
  StringPiece lbuf;
  Status s = ReadChecksummed(*offset, sizeof(uint64), &lbuf, record);
  if (!s.ok()) {
    return s;
  }
  const uint64 length = core::DecodeFixed64(lbuf.data());

  // Read data
  StringPiece data;
  s = ReadChecksummed(*offset + kHeaderSize, length, &data, record);
  if (!s.ok()) {
    if (errors::IsOutOfRange(s)) {
      s = errors::DataLoss("truncated record at ", *offset);
    }
    return s;
  }

  if (record->data() != data.data()) {
    // RandomAccessFile placed the data in some other location.
    memmove(&(*record)[0], data.data(), data.size());
  }

  record->resize(data.size());

  *offset += kHeaderSize + length + kFooterSize;
  return Status::OK();
}

}  // namespace io
}  // namespace tensorflow
