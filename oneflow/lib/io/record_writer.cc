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

#include "tensorflow/core/lib/io/record_writer.h"

#include "tensorflow/core/lib/core/coding.h"
#include "tensorflow/core/lib/hash/crc32c.h"
#include "tensorflow/core/platform/env.h"

namespace tensorflow {
namespace io {

RecordWriter::RecordWriter(WritableFile* dest,
                           const RecordWriterOptions& options)
    : dest_(dest), options_(options) {
  if (options.compression_type == RecordWriterOptions::ZLIB_COMPRESSION) {
// We don't have zlib available on all embedded platforms, so fail.
#if defined(IS_SLIM_BUILD)
    LOG(FATAL) << "Zlib compression is unsupported on mobile platforms.";
#else   // IS_SLIM_BUILD
    zlib_output_buffer_.reset(new ZlibOutputBuffer(
        dest_, options.zlib_options.input_buffer_size,
        options.zlib_options.output_buffer_size, options.zlib_options));
#endif  // IS_SLIM_BUILD
  } else if (options.compression_type == RecordWriterOptions::NONE) {
    // Nothing to do
  } else {
    LOG(FATAL) << "Unspecified compression type :" << options.compression_type;
  }
}

RecordWriter::~RecordWriter() {
#if !defined(IS_SLIM_BUILD)
  if (zlib_output_buffer_) {
    Status s = zlib_output_buffer_->Close();
    if (!s.ok()) {
      LOG(ERROR) << "Could not finish writing file: " << s;
    }
  }
#endif  // IS_SLIM_BUILD
}

static uint32 MaskedCrc(const char* data, size_t n) {
  return crc32c::Mask(crc32c::Value(data, n));
}

Status RecordWriter::WriteRecord(StringPiece data) {
  // Format of a single record:
  //  uint64    length
  //  uint32    masked crc of length
  //  byte      data[length]
  //  uint32    masked crc of data
  char header[sizeof(uint64) + sizeof(uint32)];
  core::EncodeFixed64(header + 0, data.size());
  core::EncodeFixed32(header + sizeof(uint64),
                      MaskedCrc(header, sizeof(uint64)));
  char footer[sizeof(uint32)];
  core::EncodeFixed32(footer, MaskedCrc(data.data(), data.size()));

#if !defined(IS_SLIM_BUILD)
  if (zlib_output_buffer_) {
    TF_RETURN_IF_ERROR(
        zlib_output_buffer_->Write(StringPiece(header, sizeof(header))));
    TF_RETURN_IF_ERROR(zlib_output_buffer_->Write(data));
    return zlib_output_buffer_->Write(StringPiece(footer, sizeof(footer)));
  } else {
#endif  // IS_SLIM_BUILD
    TF_RETURN_IF_ERROR(dest_->Append(StringPiece(header, sizeof(header))));
    TF_RETURN_IF_ERROR(dest_->Append(data));
    return dest_->Append(StringPiece(footer, sizeof(footer)));
#if !defined(IS_SLIM_BUILD)
  }
#endif  // IS_SLIM_BUILD
}

}  // namespace io
}  // namespace tensorflow
