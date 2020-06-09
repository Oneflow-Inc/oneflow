
/*
namespace oneflow {

namespace io {

namespace {

bool IsZlibCompressed(RecordWriterOptions options) {
  return options.compression_type == RecordWriterOptions::ZLIB_COMPRESSION;
}

}  // namespace


RecordWriterOptions RecordWriterOptions::CreateRecordWriterOptions(
    const std::string& compression_type) {
  RecordWriterOptions options;
  if (compression_type == compression::kZlib) {
    options.compression_type = io::RecordWriterOptions::ZLIB_COMPRESSION;
#if defined(IS_SLIM_BUILD)
    LOG(ERROR) << "Compression is not supported but compression_type is set."
               << " No compression will be used.";
#else
    options.zlib_options = io::ZlibCompressionOptions::DEFAULT();
#endif  // IS_SLIM_BUILD
  } else if (compression_type == compression::kGzip) {
    options.compression_type = io::RecordWriterOptions::ZLIB_COMPRESSION;
#if defined(IS_SLIM_BUILD)
    LOG(ERROR) << "Compression is not supported but compression_type is set."
               << " No compression will be used.";
#else
    options.zlib_options = io::ZlibCompressionOptions::GZIP();
#endif  // IS_SLIM_BUILD
  } else if (compression_type != compression::kNone) {
    LOG(ERROR) << "Unsupported compression_type:" << compression_type
               << ". No compression will be used.";
  }
  return options;
}


RecordWriter::RecordWriter(WritableFile* dest, const RecordWriterOptions& options)
    : dest_(dest), options_(options) {
  if (IsZlibCompressed(options)) {
// We don't have zlib available on all embedded platforms, so fail.
#if defined(IS_SLIM_BUILD)
    LOG(FATAL) << "Zlib compression is unsupported on mobile platforms.";
#else   // IS_SLIM_BUILD
    ZlibOutputBuffer* zlib_output_buffer =
        new ZlibOutputBuffer(dest, options.zlib_options.input_buffer_size,
                             options.zlib_options.output_buffer_size, options.zlib_options);
    Status s = zlib_output_buffer->Init();
    if (!s.ok()) { LOG(FATAL) << "Failed to initialize Zlib inputbuffer. Error: " << s.ToString(); }
    dest_ = zlib_output_buffer;
#endif  // IS_SLIM_BUILD
  } else if (options.compression_type == RecordWriterOptions::NONE) {
    // Nothing to do
  } else {
    LOG(FATAL) << "Unspecified compression type :" << options.compression_type;
  }
}

RecordWriter::~RecordWriter() {
  if (dest_ != nullptr) {
    Maybe<void> s = Close();
    if (!s.IsOk()) {
      LOG(ERROR) << "Could not finish writing file: " << s;
    }
  }
}

Maybe<void> RecordWriter::Close() {
  if (dest_ == nullptr) return Maybe<void>::Ok();
#if !defined(IS_SLIM_BUILD)
  if (IsZlibCompressed(options_)) {
    Maybe<void> s = dest_->Close();
    delete dest_;
    dest_ = nullptr;
    return s;
  }
#endif  // IS_SLIM_BUILD
  return Maybe<void> ::Ok();
}

Maybe<void> RecordWriter::Flush() {
  if (dest_ == nullptr) {
    return Status(::tensorflow::error::FAILED_PRECONDITION,
                  "Writer not initialized or previously closed");
  }
  return dest_->Flush();
}

}  // namespace io
}  // namespace oneflow

*/