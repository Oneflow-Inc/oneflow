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

#ifndef TENSORFLOW_LIB_IO_COMPRESSED_INPUTBUFFER_H_
#define TENSORFLOW_LIB_IO_COMPRESSED_INPUTBUFFER_H_

#include <string>
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/io/zlib_compression_options.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

// TODO(srbs|vrv): Move to a platform/zlib.h file to centralize all
// platform-specific includes
#ifdef __ANDROID__
#include "zlib.h"
#else
#include <zlib.h>
#endif  // __ANDROID__

namespace tensorflow {
namespace io {

// An ZlibInputBuffer provides support for reading from a file compressed using
// zlib (http://www.zlib.net/).
//
// A given instance of an ZlibInputBuffer is NOT safe for concurrent use
// by multiple threads
class ZlibInputBuffer {
 public:
  // Create a ZlibInputBuffer for `file` with a buffer of size
  // `input_buffer_bytes` bytes for reading contents from `file` and another
  // buffer with size `output_buffer_bytes` for caching decompressed contents.
  // Does *not* take ownership of "file".
  ZlibInputBuffer(RandomAccessFile* file, size_t input_buffer_bytes,
                  size_t output_buffer_bytes,
                  const ZlibCompressionOptions& zlib_options);

  ~ZlibInputBuffer();

  // Reads bytes_to_read bytes into *result, overwriting *result.
  //
  // Return Status codes:
  // OK:           If successful.
  // OUT_OF_RANGE: If there are not enough bytes to read before
  //               the end of the file.
  // ABORTED:      If inflate() fails, we return the error code with the
  //               error message in `z_stream_->msg`.
  // others:       If reading from file failed.
  Status ReadNBytes(int64 bytes_to_read, string* result);

 private:
  RandomAccessFile* file_;         // Not owned
  int64 file_pos_;                 // Next position to read from in `file_`
  size_t input_buffer_capacity_;   // Size of `z_stream_input_`
  size_t output_buffer_capacity_;  // Size of `z_stream_output_`
  char* next_unread_byte_;         // Next unread byte in `z_stream_output_`

  // Buffer for storing contents read from compressed file.
  // TODO(srbs): Consider using circular buffers. That would greatly simplify
  // the implementation.
  std::unique_ptr<Bytef[]> z_stream_input_;

  // Buffer for storing inflated contents of `file_`.
  std::unique_ptr<Bytef[]> z_stream_output_;

  ZlibCompressionOptions const zlib_options_;

  // Configuration passed to `inflate`.
  //
  // z_stream_->next_in:
  //   Next byte to de-compress. Points to some byte in z_stream_input_ buffer.
  // z_stream_->avail_in:
  //   Number of bytes available to be decompressed at this time.
  // z_stream_->next_out:
  //   Next byte to write de-compressed data to. Points to some byte in
  //   z_stream_output_ buffer.
  // z_stream_->avail_out:
  //   Number of free bytes available at write location.
  std::unique_ptr<z_stream> z_stream_;

  // Reads data from `file_` and tries to fill up `z_stream_input_` if enough
  // unread data is left in `file_`.
  //
  // Looks up z_stream_->next_in to check how much data in z_stream_input_
  // has already been read. The used data is removed and new data is added to
  // after any unread data in z_stream_input_.
  // After this call z_stream_->next_in points to the start of z_stream_input_
  // and z_stream_->avail_in stores the number of readable bytes in
  // z_stream_input_.
  //
  // Returns OutOfRange error if NO data could be read from file. Note that this
  // won't return an OutOfRange if there wasn't sufficient data in file to
  // completely fill up z_stream_input_.
  Status ReadFromFile();

  // Calls `inflate()` and returns DataLoss Status if it failed.
  Status Inflate();

  // Starts reading bytes at `next_unread_byte_` till either `bytes_to_read`
  // bytes have been read or `z_stream_->next_out` is reached.
  // Returns the number of bytes read and advances the `next_unread_byte_`
  // pointer to the next location to read from.
  size_t ReadBytesFromCache(size_t bytes_to_read, string* result);

  // The number of unread bytes in z_stream_output_.
  //
  // z_stream_output_  -->
  //
  // [RRRRRRRRRRRRRRRRRRUUUUUUUUUUUUUU000000000000000000]
  //                    ^             ^
  //           next_unread_byte_    z_stream_->next_out
  //
  // R: Read bytes
  // U: Unread bytes
  // 0: garbage bytes where new output will be written
  //
  // Returns the size of [next_unread_byte_, z_stream_->next_out)
  size_t NumUnreadBytes() const;

  TF_DISALLOW_COPY_AND_ASSIGN(ZlibInputBuffer);
};

}  // namespace io
}  // namespace tensorflow

#endif  // TENSORFLOW_LIB_IO_CompressedInputBuffer_H_
