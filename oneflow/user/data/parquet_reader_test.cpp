/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#ifdef __linux__

#include "parquet/api/reader.h"

#include "oneflow/core/common/maybe.h"

#include <gtest/gtest.h>
#include <unistd.h>
#include <errno.h>

namespace oneflow {

namespace test {

bool ReadFromParquet(const std::string& filename, const std::list<int>& columns,
                     bool memory_map = false, bool format_json = false, bool print_values = false,
                     bool format_dump = false, bool print_key_value_metadata = false) {
  try {
    std::unique_ptr<parquet::ParquetFileReader> reader =
        parquet::ParquetFileReader::OpenFile(filename, memory_map);
    parquet::ParquetFilePrinter printer(reader.get());
    if (format_json) {
      printer.JSONPrint(std::cout, columns, filename.c_str());
    } else {
      printer.DebugPrint(std::cout, columns, print_values, format_dump, print_key_value_metadata,
                         filename.c_str());
    }
    return true;
  } catch (const std::exception& e) {
    std::cerr << "Parquet error: " << e.what() << std::endl;
    return false;
  }
}

Maybe<std::string> GetExecutablePath() {
  static const std::string self_exe_sym_link = "/proc/self/exe";
  constexpr size_t kMaxPathLen = 1024;
  char buff[kMaxPathLen];
  ssize_t path_len = readlink(self_exe_sym_link.c_str(), buff, kMaxPathLen - 1);
  if (path_len < 0) {
    return Error::RuntimeError() << "readlink " << self_exe_sym_link
                                 << " failed with error: " << strerror(errno);
  }
  buff[path_len] = '\0';
  return std::string(buff);
}

Maybe<std::string> GetDirName(const std::string& path) {
  size_t pos = path.find_last_of("/");
  return path.substr(0, pos + 1);
}

TEST(ParquetReader, read_and_print) {
  // std::string parquet_example_file = *CHECK_JUST(GetExecutablePath());
  // parquet_example_file = *CHECK_JUST(GetDirName(parquet_example_file));
  // parquet_example_file += "../external/arrow/arrow-src/python/pyarrow/tests/"
  //                         "data/parquet/v0.7.1.some-named-index.parquet";
  std::string parquet_example_file =
      "/dataset/wdl_parquet/train/"
      "part-00000-6e6f50b9-75e8-4917-87ae-85351fe65d36-c000.snappy.parquet";
  ASSERT_TRUE(ReadFromParquet(parquet_example_file, {0, 1, 2, 3}, /*memory_map*/ true,
                              /*format_json*/ false, /*print_values*/ true,
                              /*format_dump*/ true, /*print_key_value_metadata*/ true));
}

}  // namespace test

}  // namespace oneflow

#endif  // __linux__
