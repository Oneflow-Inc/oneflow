#include "oneflow/core/common/str_util.h"
#include "glog/logging.h"
#include "gtest/gtest.h"
#include "tensorflow/core/lib/io/path.h"

namespace oneflow {

void testUri(std::string uri) {
  std::string dir = tensorflow::io::Dirname(uri).ToString();
  LOG(INFO) << "dirname$" << dir;
  tensorflow::StringPiece schema, host, path;
  tensorflow::io::ParseURI(uri, &schema, &host, &path);
  LOG(INFO) << "schema$" << schema.ToString();
  LOG(INFO) << "host$" << host.ToString();
  LOG(INFO) << "path$" << path.ToString();
  ASSERT_EQ(dir, Dirname(uri));
}

TEST(str_util, parseUri) {
  std::string uri = "://127.0.0.1:8080/path/to";
  // std::string uri_dirname = "://127.0.0.1:8080/path";
  testUri("://127.0.0.1:8080/path/to");
  testUri("://127.0.0.1:8080/path/to/2/");
  testUri("path/to");
  testUri("/asdgasdgasdg");
  testUri("hdfs://127.0.0.1:8080/path/to");
  testUri("C:/path/to");
  testUri("asdgasdgasdg");
  testUri("asdgasdgasdg/");
}

}  // namespace oneflow
