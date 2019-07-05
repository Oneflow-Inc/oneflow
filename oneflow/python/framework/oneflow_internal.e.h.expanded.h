# 1 "/home/lixinqi/oneflow/oneflow/python/framework/oneflow_internal.e.h"
# 1 "<built-in>"
# 1 "<command-line>"
# 1 "/usr/include/stdc-predef.h" 1 3 4
# 1 "<command-line>" 2
# 1 "/home/lixinqi/oneflow/oneflow/python/framework/oneflow_internal.e.h"
# 1 "/home/lixinqi/oneflow/oneflow/core/common/preprocessor.h" 1

# 1 "/home/lixinqi/oneflow/oneflow/core/common/preprocessor_internal.h" 1
# 5 "/home/lixinqi/oneflow/oneflow/core/common/preprocessor.h" 2
# 2 "/home/lixinqi/oneflow/oneflow/python/framework/oneflow_internal.e.h" 2
# 1 "/home/lixinqi/oneflow/oneflow/core/common/data_type_seq.h" 1
# 3 "/home/lixinqi/oneflow/oneflow/python/framework/oneflow_internal.e.h" 2
# 16 "/home/lixinqi/oneflow/oneflow/python/framework/oneflow_internal.e.h"
void OfBlob_CopyToBuffer_float(uint64_t of_blob_ptr, float* ptr, int64_t len) {
  using namespace oneflow;
  auto* of_blob = reinterpret_cast<OfBlob*>(of_blob_ptr);
  of_blob->AutoMemCopyTo<float>(ptr, len);
}
void OfBlob_CopyFromBuffer_float(uint64_t of_blob_ptr, const float* ptr, int64_t len) {
  using namespace oneflow;
  auto* of_blob = reinterpret_cast<OfBlob*>(of_blob_ptr);
  of_blob->AutoMemCopyFrom<float>(ptr, len);
}
void OfBlob_CopyToBuffer_double(uint64_t of_blob_ptr, double* ptr, int64_t len) {
  using namespace oneflow;
  auto* of_blob = reinterpret_cast<OfBlob*>(of_blob_ptr);
  of_blob->AutoMemCopyTo<double>(ptr, len);
}
void OfBlob_CopyFromBuffer_double(uint64_t of_blob_ptr, const double* ptr, int64_t len) {
  using namespace oneflow;
  auto* of_blob = reinterpret_cast<OfBlob*>(of_blob_ptr);
  of_blob->AutoMemCopyFrom<double>(ptr, len);
}
void OfBlob_CopyToBuffer_int8_t(uint64_t of_blob_ptr, int8_t* ptr, int64_t len) {
  using namespace oneflow;
  auto* of_blob = reinterpret_cast<OfBlob*>(of_blob_ptr);
  of_blob->AutoMemCopyTo<int8_t>(ptr, len);
}
void OfBlob_CopyFromBuffer_int8_t(uint64_t of_blob_ptr, const int8_t* ptr, int64_t len) {
  using namespace oneflow;
  auto* of_blob = reinterpret_cast<OfBlob*>(of_blob_ptr);
  of_blob->AutoMemCopyFrom<int8_t>(ptr, len);
}
void OfBlob_CopyToBuffer_int32_t(uint64_t of_blob_ptr, int32_t* ptr, int64_t len) {
  using namespace oneflow;
  auto* of_blob = reinterpret_cast<OfBlob*>(of_blob_ptr);
  of_blob->AutoMemCopyTo<int32_t>(ptr, len);
}
void OfBlob_CopyFromBuffer_int32_t(uint64_t of_blob_ptr, const int32_t* ptr, int64_t len) {
  using namespace oneflow;
  auto* of_blob = reinterpret_cast<OfBlob*>(of_blob_ptr);
  of_blob->AutoMemCopyFrom<int32_t>(ptr, len);
}
void OfBlob_CopyToBuffer_int64_t(uint64_t of_blob_ptr, int64_t* ptr, int64_t len) {
  using namespace oneflow;
  auto* of_blob = reinterpret_cast<OfBlob*>(of_blob_ptr);
  of_blob->AutoMemCopyTo<int64_t>(ptr, len);
}
void OfBlob_CopyFromBuffer_int64_t(uint64_t of_blob_ptr, const int64_t* ptr, int64_t len) {
  using namespace oneflow;
  auto* of_blob = reinterpret_cast<OfBlob*>(of_blob_ptr);
  of_blob->AutoMemCopyFrom<int64_t>(ptr, len);
}
void OfBlob_CopyToBuffer_char(uint64_t of_blob_ptr, char* ptr, int64_t len) {
  using namespace oneflow;
  auto* of_blob = reinterpret_cast<OfBlob*>(of_blob_ptr);
  of_blob->AutoMemCopyTo<char>(ptr, len);
}
void OfBlob_CopyFromBuffer_char(uint64_t of_blob_ptr, const char* ptr, int64_t len) {
  using namespace oneflow;
  auto* of_blob = reinterpret_cast<OfBlob*>(of_blob_ptr);
  of_blob->AutoMemCopyFrom<char>(ptr, len);
}
void OfBlob_CopyToBuffer_uint8_t(uint64_t of_blob_ptr, uint8_t* ptr, int64_t len) {
  using namespace oneflow;
  auto* of_blob = reinterpret_cast<OfBlob*>(of_blob_ptr);
  of_blob->AutoMemCopyTo<uint8_t>(ptr, len);
}
void OfBlob_CopyFromBuffer_uint8_t(uint64_t of_blob_ptr, const uint8_t* ptr, int64_t len) {
  using namespace oneflow;
  auto* of_blob = reinterpret_cast<OfBlob*>(of_blob_ptr);
  of_blob->AutoMemCopyFrom<uint8_t>(ptr, len);
};

std::string OfBlob_GetCopyToBufferFuncName(uint64_t of_blob_ptr) {
  using namespace oneflow;
  static const HashMap<int64_t, std::string> data_type2func_name{

      {DataType::kFloat, "OfBlob_CopyToBuffer_"
                         "float"},
      {DataType::kDouble, "OfBlob_CopyToBuffer_"
                          "double"},
      {DataType::kInt8, "OfBlob_CopyToBuffer_"
                        "int8_t"},
      {DataType::kInt32, "OfBlob_CopyToBuffer_"
                         "int32_t"},
      {DataType::kInt64, "OfBlob_CopyToBuffer_"
                         "int64_t"},
      {DataType::kChar, "OfBlob_CopyToBuffer_"
                        "char"},
      {DataType::kUInt8, "OfBlob_CopyToBuffer_"
                         "uint8_t"},

  };
  auto* of_blob = reinterpret_cast<OfBlob*>(of_blob_ptr);
  return data_type2func_name.at(of_blob->data_type());
}

std::string OfBlob_GetCopyFromBufferFuncName(uint64_t of_blob_ptr) {
  using namespace oneflow;
  static const HashMap<int64_t, std::string> data_type2func_name{

      {DataType::kFloat, "OfBlob_CopyFromBuffer_"
                         "float"},
      {DataType::kDouble, "OfBlob_CopyFromBuffer_"
                          "double"},
      {DataType::kInt8, "OfBlob_CopyFromBuffer_"
                        "int8_t"},
      {DataType::kInt32, "OfBlob_CopyFromBuffer_"
                         "int32_t"},
      {DataType::kInt64, "OfBlob_CopyFromBuffer_"
                         "int64_t"},
      {DataType::kChar, "OfBlob_CopyFromBuffer_"
                        "char"},
      {DataType::kUInt8, "OfBlob_CopyFromBuffer_"
                         "uint8_t"},

  };
  auto* of_blob = reinterpret_cast<OfBlob*>(of_blob_ptr);
  return data_type2func_name.at(of_blob->data_type());
}
