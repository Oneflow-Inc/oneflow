#include "oneflow/core/schedule/sxml.h"
#include "gtest/gtest.h"

namespace oneflow {
namespace schedule {

TEST(SXML, plain_text) {
  SXML doc("", "good");
  ASSERT_EQ(doc.ToString(), "good");
}

TEST(SXML, tag_simple) {
  SXML doc("br");
  ASSERT_EQ(doc.ToString(), "<br></br>");
}

TEST(SXML, tree) {
  SXML doc("p", {{"br", ""}, {"", "good"}});
  ASSERT_EQ(doc.ToString(), "<p>\n <br></br>\n good\n</p>");
}

}  // namespace schedule
}  // namespace oneflow
