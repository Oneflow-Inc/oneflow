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

#ifndef TENSORFLOW_LIB_IO_RECORD_WRITER_H_
#define TENSORFLOW_LIB_IO_RECORD_WRITER_H_

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#if !defined(IS_SLIM_BUILD)
#include "tensorflow/core/lib/io/zlib_compression_options.h"
#include "tensorflow/core/lib/io/zlib_outputbuffer.h"
#endif  // IS_SLIM_BUILD
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

class WritableFile;

namespace io {

class RecordWriterOptions {
 public:
  enum CompressionType { NONE = 0, ZLIB_COMPRESSION = 1 };
  CompressionType compression_type = NONE;

// Options specific to zlib compression.
#if !defined(IS_SLIM_BUILD)
  ZlibCompressionOptions zlib_options;
#endif  // IS_SLIM_BUILD
};

class RecordWriter {
 public:
  // Create a writer that will append data to "*dest".
  // "*dest" must be initially empty.
  // "*dest" must remain live while this Writer is in use.
  RecordWriter(WritableFile* dest,
               const RecordWriterOptions& options = RecordWriterOptions());

  ~RecordWriter();

  Status WriteRecord(StringPiece slice);

  // Flushes any buffered data held by underlying containers of the
  // RecordWriter to the WritableFile. Does *not* flush the
  // WritableFile.
  Status Flush() {
#if !defined(IS_SLIM_BUILD)
    if (zlib_output_buffer_) {
      return zlib_output_buffer_->Flush();
    }
#endif  // IS_SLIM_BUILD

    return Status::OK();
  }

 private:
  WritableFile* const dest_;
  RecordWriterOptions options_;
#if !defined(IS_SLIM_BUILD)
  std::unique_ptr<ZlibOutputBuffer> zlib_output_buffer_;
#endif  // IS_SLIM_BUILD

  TF_DISALLOW_COPY_AND_ASSIGN(RecordWriter);
};

}  // namespace io
}  // namespace tensorflow

#endif  // TENSORFLOW_LIB_IO_RECORD_WRITER_H_
