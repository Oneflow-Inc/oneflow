/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef THIRD_PARTY_TENSORFLOW_CORE_LIB_IO_COMPRESSED_OUTPUTBUFFER_H_
#define THIRD_PARTY_TENSORFLOW_CORE_LIB_IO_COMPRESSED_OUTPUTBUFFER_H_

#include <string>
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/io/zlib_compression_options.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

// TODO(srbs|vrv): Move to a platform/zlib.h file to centralize all
// platform-specific includes.
#ifdef __ANDROID__
#include "zlib.h"
#else
#include <zlib.h>
#endif  // __ANDROID__

namespace tensorflow {
namespace io {

// Provides support for writing compressed output to file using zlib
// (http://www.zlib.net/).
// A given instance of an ZlibOutputBuffer is NOT safe for concurrent use
// by multiple threads
class ZlibOutputBuffer {
 public:
  // Create an ZlibOutputBuffer for `file` with two buffers that cache the
  // 1. input data to be deflated
  // 2. the deflated output
  // with sizes `input_buffer_bytes` and `output_buffer_bytes` respectively.
  // Does not take ownership of `file`.
  ZlibOutputBuffer(
      WritableFile* file,
      int32 input_buffer_bytes,   // size of z_stream.next_in buffer
      int32 output_buffer_bytes,  // size of z_stream.next_out buffer
      const ZlibCompressionOptions& zlib_options);

  ~ZlibOutputBuffer();

  // Adds `data` to the compression pipeline.
  //
  // The input data is buffered in `z_stream_input_` and is compressed in bulk
  // when the buffer gets full. The compressed output is not immediately
  // written to file but rather buffered in `z_stream_output_` and gets written
  // to file when the buffer is full.
  //
  // To immediately write contents to file call `Flush()`.
  Status Write(StringPiece data);

  // Deflates any cached input and writes all output to file.
  Status Flush();

  // Compresses any cached input and writes all output to file. This must be
  // called before the destructor to avoid any data loss.
  //
  // Contrary to `Flush()` this informs zlib that it should not expect any
  // further input by using Z_FINISH flush mode. Also cleans up z_stream.
  //
  // After calling this, any further calls to `Write()`, `Flush()` or `Close()`
  // will fail.
  Status Close();

 private:
  WritableFile* file_;  // Not owned
  size_t input_buffer_capacity_;
  size_t output_buffer_capacity_;

  // Buffer for storing contents read from input `file_`.
  // TODO(srbs): Consider using circular buffers. That would greatly simplify
  // the implementation.
  std::unique_ptr<Bytef[]> z_stream_input_;

  // Buffer for storing deflated contents of `file_`.
  std::unique_ptr<Bytef[]> z_stream_output_;

  ZlibCompressionOptions const zlib_options_;

  // Configuration passed to `deflate`.
  //
  // z_stream_->next_in:
  //   Next byte to compress. Points to some byte in z_stream_input_ buffer.
  // z_stream_->avail_in:
  //   Number of bytes available to be compressed at this time.
  // z_stream_->next_out:
  //   Next byte to write compressed data to. Points to some byte in
  //   z_stream_output_ buffer.
  // z_stream_->avail_out:
  //   Number of free bytes available at write location.
  std::unique_ptr<z_stream> z_stream_;

  // Adds `data` to `z_stream_input_`.
  // Throws if `data.size()` > AvailableInputSpace().
  void AddToInputBuffer(StringPiece data);

  // Returns the total space available in z_input_stream_ buffer.
  int32 AvailableInputSpace() const;

  // Deflate contents in z_stream_input_ and store results in z_stream_output_.
  // The contents of output stream are written to file if more space is needed.
  // On successful termination it is assured that:
  // - z_stream_->avail_in == 0
  // - z_stream_->avail_out > 0
  //
  // Note: This method does not flush contents to file.
  // Returns non-ok status if writing contents to file fails.
  Status DeflateBuffered(bool last = false);

  // Appends contents of `z_stream_output_` to `file_`.
  // Returns non-OK status if writing to file fails.
  Status FlushOutputBufferToFile();

  // Calls `deflate()` and returns DataLoss Status if it failed.
  Status Deflate(int flush);

  static bool IsSyncOrFullFlush(uint8 flush_mode) {
    return flush_mode == Z_SYNC_FLUSH || flush_mode == Z_FULL_FLUSH;
  }

  TF_DISALLOW_COPY_AND_ASSIGN(ZlibOutputBuffer);
};

}  // namespace io
}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CORE_LIB_IO_COMPRESSED_OUTPUTBUFFER_H_
