#include "oneflow/core/common/util.h"
#include "oneflow/core/common/zero_only_zip.h"

namespace oneflow {

// test case for "\0\0\0\0\0\0\0\234\0\0\0\0\0" -> -73234-5
TEST(ZipOnlyZipUtil, zip1) {
  const char data[] = {0x00,0x00,0x00,0x00,0x00,0x00,0x00,'2','3','4',0x00,0x00,0x00,0x00,0x00};
  char buffer[4096 + sizeof(SizedBufferView)];
  SizedBufferView *sized_buffer = SizedBufferView::PlacementNew(buffer, sizeof(SizedBufferView));
  ZeroOnlyZipUtil zero_zip;
  zero_zip.ZipToSizedBuffer(data, sizeof(data), sized_buffer);
  ASSERT_EQ(int(sized_buffer -> data[0]), -7);
  ASSERT_EQ(int(sized_buffer -> data[1]), 3);
  ASSERT_EQ(sized_buffer -> data[2], '2');
  ASSERT_EQ(sized_buffer -> data[3], '3');
  ASSERT_EQ(sized_buffer -> data[4], '4');
  ASSERT_EQ(int(sized_buffer -> data[5]), -5);
}


// test case for "abcde\0\0\0\0\0\0\0234" -> 5abcde-73234
TEST(ZipOnlyZipUtil, zip2) {
  const char data[] = {'a','b','c','d','e',0x00,0x00,0x00,0x00,0x00,0x00,0x00,'2','3','4'};
  char buffer[4096 + sizeof(SizedBufferView)];
  SizedBufferView *sized_buffer = SizedBufferView::PlacementNew(buffer, sizeof(SizedBufferView));
  ZeroOnlyZipUtil zero_zip;
  zero_zip.ZipToSizedBuffer(data, sizeof(data), sized_buffer);
  ASSERT_EQ(int(sized_buffer -> data[0]), 5);
  ASSERT_EQ(sized_buffer -> data[1], 'a');
  ASSERT_EQ(sized_buffer -> data[2], 'b');
  ASSERT_EQ(sized_buffer -> data[3], 'c');
  ASSERT_EQ(sized_buffer -> data[4], 'd');
  ASSERT_EQ(sized_buffer -> data[5], 'e');
  ASSERT_EQ(int(sized_buffer -> data[6]), -7);
  ASSERT_EQ(int(sized_buffer -> data[7]), 3);
  ASSERT_EQ(sized_buffer -> data[8], '2');
  ASSERT_EQ(sized_buffer -> data[9], '3');
  ASSERT_EQ(sized_buffer -> data[10], '4');
}


// test case for "\0\0\0\0\0\0\0\0\0\0" -> -10
TEST(ZipOnlyZipUtil, zip3)
  const char data[] = {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00};
  char buffer[4096 + sizeof(SizedBufferView)];
  SizedBufferView *sized_buffer = SizedBufferView::PlacementNew(buffer, sizeof(SizedBufferView));
  ZeroOnlyZipUtil zero_zip;
  zero_zip.ZipToSizedBuffer(data, sizeof(data), sized_buffer);
  ASSERT_EQ(int(sized_buffer -> data[0]), -10);
}


// test case for -73234-5 -> "\0\0\0\0\0\0\0234\0\0\0\0\0"
TEST(ZipOnlyZipUtil, unzip1) {
  char buffer[4096 + sizeof(SizedBufferView)];
  SizedBufferView *sizedbuffer = SizedBufferView::PlacementNew(buffer, sizeof(SizedBufferView));
  sizedbuffer -> data[0] = char(0xf9);
  sizedbuffer -> data[1] = char(3);
  sizedbuffer -> data[2] = '2';
  sizedbuffer -> data[3] = '3';
  sizedbuffer -> data[4] = '4';
  sizedbuffer -> data[5] = char(0xfa);
  char *data = new char[15];  // actually, new char[1] also works.
  ZeroOnlyZipUtil zero_unzip;
  zero_unzip.UnzipToExpectedSize(*sizedbuffer, data, 15);
  ASSERT_EQ(data[0], 0x00);
  ASSERT_EQ(data[1], 0x00);
  ASSERT_EQ(data[2], 0x00);
  ASSERT_EQ(data[3], 0x00);
  ASSERT_EQ(data[4], 0x00);
  ASSERT_EQ(data[5], 0x00);
  ASSERT_EQ(data[6], 0x00);
  ASSERT_EQ(data[7], '2');
  ASSERT_EQ(data[8], '3');
  ASSERT_EQ(data[9], '4');
  ASSERT_EQ(data[10], 0x00);
  ASSERT_EQ(data[11], 0x00);
  ASSERT_EQ(data[12], 0x00);
  ASSERT_EQ(data[13], 0x00);
  ASSERT_EQ(data[14], 0x00);
  delete[] data;
}


// test case for 5abcde-73234 -> "abcde\0\0\0\0\0\0\0234"
TEST(ZipOnlyZipUtil, unzip2) {
  char buffer[4096 + sizeof(SizedBufferView)];
  SizedBufferView *sizedbuffer = SizedBufferView::PlacementNew(buffer, sizeof(SizedBufferView));
  sizedbuffer -> data[0] = char(5);
  sizedbuffer -> data[1] = 'a';
  sizedbuffer -> data[2] = 'b';
  sizedbuffer -> data[3] = 'c';
  sizedbuffer -> data[4] = 'd';
  sizedbuffer -> data[5] = 'e'
  sizedbuffer -> data[6] = char(0xf9);
  sizedbuffer -> data[7] = char(3);
  sizedbuffer -> data[8] = '2';
  sizedbuffer -> data[9] = '3';
  sizedbuffer -> data[10] = '4';
  char *data = new char[15];  // actually, new char[1] also works.
  ZeroOnlyZipUtil zero_unzip;
  zero_unzip.UnzipToExpectedSize(*sizedbuffer, data, 15);
  ASSERT_EQ(data[0], 'a');
  ASSERT_EQ(data[1], 'b');
  ASSERT_EQ(data[2], 'c');
  ASSERT_EQ(data[3], 'd');
  ASSERT_EQ(data[4], 'e');
  ASSERT_EQ(data[5], 0x00);
  ASSERT_EQ(data[6], 0x00);
  ASSERT_EQ(data[7], 0x00);
  ASSERT_EQ(data[8], 0x00);
  ASSERT_EQ(data[9], 0x00);
  ASSERT_EQ(data[10], 0x00);
  ASSERT_EQ(data[11], 0x00);
  ASSERT_EQ(data[12], '2');
  ASSERT_EQ(data[13], '3');
  ASSERT_EQ(data[14], '4');
  delete[] data;
}

}  // namespace oneflow
