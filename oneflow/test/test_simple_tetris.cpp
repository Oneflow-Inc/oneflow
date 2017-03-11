#include <gtest/gtest.h>

#include "task/fsm/tetris.h"

// simple test for simple tetris
namespace caffe {

class TetrisTest :public testing::Test {
 protected:
   virtual void SetUp() {}
   Tetris<int64_t> tetris;
};

TEST_F(TetrisTest, AllTetrisTest) {
  std::shared_ptr<TetrisIDColumn<int64_t>> c1(new
    TetrisIDColumn<int64_t>(0, 3));
  EXPECT_EQ(c1->Ready(), false);
  std::shared_ptr<TetrisIDColumn<int64_t>> c2(new
    TetrisIDColumn<int64_t>(1, 2));
  std::shared_ptr<TetrisFIFOColumn<int64_t>> c3(new
    TetrisFIFOColumn<int64_t>(2));
  EXPECT_EQ(c3->Ready(), false);
  std::shared_ptr<TetrisFIFOColumn<int64_t>> c4(new
    TetrisFIFOColumn<int64_t>(3));

  tetris.Add(c1);
  tetris.Add(c2);
  tetris.Add(c3);
  tetris.Add(c4);

  //auto c1 = tetris.GetColumnByName(0);
  //EXPECT_EQ(c1->name(), 0);
  //EXPECT_EQ(c1->Ready(), false);
  //auto c2 = tetris.GetColumnByName(1);
  //EXPECT_EQ(c2->name(), 1);
  //EXPECT_EQ(c2->Ready(), false);
  //auto c3 = tetris.GetColumnByName(2);
  //EXPECT_EQ(c3->name(), 2);
  //EXPECT_EQ(c3->Ready(), false);
  //auto c4 = tetris.GetColumnByName(3);
  //EXPECT_EQ(c4->name(), 3);
  //EXPECT_EQ(c4->Ready(), false);

  c1->Push(10, 1);
  c1->Push(11, 1);
  EXPECT_EQ(c1->Ready(), false);
  c1->Push(12, 1);
  EXPECT_EQ(c1->Ready(), true);
  auto items = c1->Pop();
  EXPECT_EQ(c1->Ready(), false);
  EXPECT_EQ(items[0].first, 10);
  EXPECT_EQ(items[1].first, 11);
  EXPECT_EQ(items[2].first, 12);
  EXPECT_EQ(items[0].second, 1);
  EXPECT_EQ(items[1].second, 1);
  EXPECT_EQ(items[2].second, 1);

  c3->Push(10, 1);
  c3->Push(11, 2);
  EXPECT_EQ(c3->Ready(), true);
  items = c3->Pop();
  EXPECT_EQ(c3->Ready(), true);
  EXPECT_EQ(items[0].first, 10);
  EXPECT_EQ(items[0].second, 1);
  items = c3->Pop();
  EXPECT_EQ(c3->Ready(), false);
  EXPECT_EQ(items[0].first, 11);
  EXPECT_EQ(items[0].second, 2);


  // now test  Tetris
  tetris.Push(0, 10, 1);
  tetris.Push(0, 11, 1);
  tetris.Push(0, 12, 2);
  tetris.Push(0, 13, 2);
  tetris.Push(0, 14, 2);  // ready id 2
  EXPECT_EQ(tetris.Ready(), false);

  tetris.Push(1, 110, 1);
  tetris.Push(1, 112, 3);
  tetris.Push(1, 114, 4);
  tetris.Push(1, 115, 4);
  tetris.Push(1, 113, 3);
  tetris.Push(1, 111, 1);  // ready id 4, 3, 1
  EXPECT_EQ(tetris.Ready(), false);

  tetris.Push(2, 1000, -1);
  tetris.Push(2, 1001, -1);
  tetris.Push(2, 1002, -1);
  EXPECT_EQ(tetris.Ready(), false);

  tetris.Push(3, 10000, 8);
  EXPECT_EQ(tetris.Ready(), true);

  auto tetris_items = tetris.Pop();

  EXPECT_EQ(tetris.Ready(), false);
  EXPECT_EQ(c1->Ready(), false);
  EXPECT_EQ(c2->Ready(), true);
  EXPECT_EQ(c3->Ready(), true);
  EXPECT_EQ(c4->Ready(), false);

  auto c1_kv = tetris_items.find(0);
  EXPECT_NE(c1_kv, tetris_items.end());
  items = c1_kv->second;
  EXPECT_EQ(items.size(), 3);
  EXPECT_EQ(items[0].first, 12);
  EXPECT_EQ(items[1].first, 13);
  EXPECT_EQ(items[2].first, 14);
  EXPECT_EQ(items[0].second, 2);
  EXPECT_EQ(items[1].second, 2);
  EXPECT_EQ(items[2].second, 2);

  auto c2_kv = tetris_items.find(1);
  EXPECT_NE(c2_kv, tetris_items.end());
  items = c2_kv->second;
  EXPECT_EQ(items.size(), 2);
  EXPECT_EQ(items[0].first, 114);
  EXPECT_EQ(items[1].first, 115);
  EXPECT_EQ(items[0].second, 4);
  EXPECT_EQ(items[1].second, 4);

  auto c3_kv = tetris_items.find(2);
  EXPECT_NE(c3_kv, tetris_items.end());
  items = c3_kv->second;
  EXPECT_EQ(items.size(), 1);
  EXPECT_EQ(items[0].first, 1000);
  EXPECT_EQ(items[0].second, -1);

  auto c4_kv = tetris_items.find(3);
  EXPECT_NE(c4_kv, tetris_items.end());
  items = c4_kv->second;
  EXPECT_EQ(items.size(), 1);
  EXPECT_EQ(items[0].first, 10000);
  EXPECT_EQ(items[0].second, 8);
}

}  // namespace caffe
