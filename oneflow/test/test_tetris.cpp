#include <gtest/gtest.h>

#include "common/composite_tetris.h"

// simple test for tetris
namespace caffe {

class TetrisTest :public testing::Test {
 protected:
   virtual void SetUp() {
     tetris.set_name("test_tetris");
     column.set_name("test_column");
     id_tetris.set_name("test_id_tetris");
   }
   FIFOTetris<int64_t> tetris;
   TetrisColumn<int64_t> column;
   IDTetris<int64_t> id_tetris;
};

TEST_F(TetrisTest, ColumnTest) {
  EXPECT_EQ(column.type(), TetrisType::kColumn);
  EXPECT_EQ(column.HasReady(), false);
  column.Push(11);
  column.Push(22);
  column.Push(33);
  column.Push(10, 1);
  column.Push(20, 2);
  column.Push(30, 3);
  auto rs = column.Pop();
  EXPECT_EQ(rs["test_column"].second, 11);
  rs = column.Pop();
  EXPECT_EQ(rs["test_column"].second, 22);
  rs = column.PopDataOfID(2);
  EXPECT_EQ(rs["test_column"].first, 2);
  EXPECT_EQ(rs["test_column"].second, 20);
  std::multiset<int64_t> ready_ids = {-1, 1, 3 };
  EXPECT_EQ(column.GetReadyIDs(), ready_ids);
  EXPECT_EQ(column.HasReady(), true);
}

TEST_F(TetrisTest, IDTetrisTest) {
  EXPECT_EQ(id_tetris.type(), TetrisType::kIDTetris);
  EXPECT_EQ(id_tetris.HasReady(), false);
  std::shared_ptr<TetrisColumn<int64_t>> id_col_1(new TetrisColumn<int64_t>("id_col_1"));
  id_tetris.add(id_col_1);
  EXPECT_EQ(id_tetris.HasReady(), false);
  EXPECT_EQ(id_tetris.GetTetrisByName("id_col_1"), id_col_1);
  std::multiset<int64_t> ready_ids;
  EXPECT_EQ(id_tetris.GetReadyIDs(), ready_ids);

  id_col_1->Push(98);
  id_col_1->Push(99, 4);
  id_col_1->Push(100, 9);
  EXPECT_EQ(id_tetris.HasReady(), true);
  ready_ids = { -1, 4, 9 };
  EXPECT_EQ(id_tetris.GetReadyIDs(), ready_ids);
  auto rs = id_tetris.Pop();
  EXPECT_EQ(rs["id_col_1"].second, 98);
  EXPECT_EQ(rs["id_col_1"].first, -1);
  rs = id_tetris.PopDataOfID(9);
  EXPECT_EQ(rs["id_col_1"].second, 100);
  EXPECT_EQ(rs["id_col_1"].first, 9);
  ready_ids = { 4 };
  EXPECT_EQ(id_tetris.GetReadyIDs(), ready_ids);
  rs = id_tetris.Pop();
  EXPECT_EQ(id_tetris.HasReady(), false);

  // 3 level, 3 tetris, 5 columns in total:
  // test_tetris contains: column id_col_1, id_col_2, IDTetris id_t_1.
  // id_t_1 contains: column id_t_1_c_1, IDTetris id_t_2.
  // id_t_2 contains: column id_t_2_c_1, id_t_2_c_2.
  // 
  std::shared_ptr<IDTetris<int64_t>> id_t_1(new IDTetris<int64_t>("id_t_1"));
  id_tetris.add(id_t_1);  
  std::shared_ptr<IDTetris<int64_t>> id_t_2(new IDTetris<int64_t>("id_t_2"));
  id_t_1->add(id_t_2);
  std::shared_ptr<TetrisColumn<int64_t>> id_t_1_c_1(new TetrisColumn<int64_t>("id_t_1_c_1"));
  id_t_1->add(id_t_1_c_1);
  std::shared_ptr<TetrisColumn<int64_t>> id_t_2_c_1(new TetrisColumn<int64_t>("id_t_2_c_1"));
  id_t_2->add(id_t_2_c_1);
  std::shared_ptr<TetrisColumn<int64_t>> id_t_2_c_2(new TetrisColumn<int64_t>("id_t_2_c_2"));
  id_t_2->add(id_t_2_c_2);

  std::shared_ptr<TetrisColumn<int64_t>> id_col_2(new TetrisColumn<int64_t>("id_col_2"));
  id_tetris.add(id_col_2);

  // test id_t_2
  id_t_2_c_1->Push(21, 2);
  id_t_2_c_1->Push(32, 4);
  id_t_2_c_2->Push(3, 4);
  id_t_2_c_2->Push(4, 5);
  id_t_2_c_2->Push(5, 6);
  EXPECT_EQ(id_t_2->HasReady(), true);
  ready_ids = { 4 };
  EXPECT_EQ(id_t_2->GetReadyIDs(), ready_ids);
  rs = id_t_2->Pop();
  EXPECT_EQ(rs["id_t_2_c_1"].first, std::make_pair(4, 32).first);
  EXPECT_EQ(rs["id_t_2_c_1"].second, std::make_pair(4, 32).second);
  EXPECT_EQ(rs["id_t_2_c_2"].first, std::make_pair(4, 3).first);
  EXPECT_EQ(rs["id_t_2_c_2"].second, std::make_pair(4, 3).second);
  EXPECT_EQ(id_t_2->HasReady(), false);
  ready_ids.clear();
  EXPECT_EQ(id_t_2->GetReadyIDs(), ready_ids);

  id_t_2_c_1->Push(51, 5);
  id_t_2_c_1->Push(32, 4);
  id_t_2_c_1->Push(328, 8);
  id_t_2_c_1->Push(329, 9);

  id_t_2_c_2->Push(3, 4);
  id_t_2_c_2->Push(4, 2);
  id_t_2_c_2->Push(5, 6);
  id_t_2_c_2->Push(1328, 8);
  id_t_2_c_2->Push(1329, 9);
  // common: 2, 5, 4, 8, 9

  id_t_1_c_1->Push(13);
  id_t_1_c_1->Push(14, 2);
  id_t_1_c_1->Push(15, 5);
  id_t_1_c_1->Push(16, 9);
  id_t_1_c_1->Push(17, 8);
  // common: 2, 5, 8, 9

  id_col_1->Push(11);
  id_col_1->Push(114, 2);
  id_col_1->Push(115, 5);
  id_col_1->Push(117, 8);
  // common: 2, 5, 8

  id_col_2->Push(110);
  id_col_2->Push(1150, 5);
  id_col_2->Push(1170, 8);
  // common: 5, 8

  EXPECT_EQ(id_tetris.HasReady(), true);
  ready_ids = { 5, 8 };
  EXPECT_EQ(id_tetris.GetReadyIDs(), ready_ids);

  rs = id_tetris.Pop(); // 5
  EXPECT_EQ(rs["id_t_2_c_1"].first, std::make_pair(5, 51).first);
  EXPECT_EQ(rs["id_t_2_c_1"].second, std::make_pair(5, 51).second);

  EXPECT_EQ(rs["id_t_2_c_2"].first, std::make_pair(5, 4).first);
  EXPECT_EQ(rs["id_t_2_c_2"].second, std::make_pair(5, 4).second);

  EXPECT_EQ(rs["id_t_1_c_1"].first, std::make_pair(5, 15).first);
  EXPECT_EQ(rs["id_t_1_c_1"].second, std::make_pair(5, 15).second);

  EXPECT_EQ(rs["id_col_1"].first, std::make_pair(5, 115).first);
  EXPECT_EQ(rs["id_col_1"].second, std::make_pair(5, 115).second);

  EXPECT_EQ(rs["id_col_2"].first, std::make_pair(5, 1150).first);
  EXPECT_EQ(rs["id_col_2"].second, std::make_pair(5, 1150).second);

  EXPECT_EQ(id_tetris.HasReady(), true);
  ready_ids = { 8 };
  EXPECT_EQ(id_tetris.GetReadyIDs(), ready_ids);

  rs = id_tetris.Pop(); // 8
  EXPECT_EQ(id_tetris.HasReady(), false);
  ready_ids.clear();
  EXPECT_EQ(id_tetris.GetReadyIDs(), ready_ids);
}

TEST_F(TetrisTest, FIFOTetrisTest) {
  EXPECT_EQ(tetris.type(), TetrisType::kFIFOTetris);
  EXPECT_EQ(tetris.HasReady(), false);
  // 3 level, 3 tetris, 4 columns
  // test_column: column c1, c2, FIFOTetris t1
  // t1: column c3, IDTetris t2
  // t2: column c4
  std::shared_ptr<TetrisColumn<int64_t>> c1(new TetrisColumn<int64_t>("c1"));
  std::shared_ptr<TetrisColumn<int64_t>> c2(new TetrisColumn<int64_t>("c2"));
  std::shared_ptr<TetrisColumn<int64_t>> c3(new TetrisColumn<int64_t>("c3"));
  std::shared_ptr<TetrisColumn<int64_t>> c4(new TetrisColumn<int64_t>("c4"));
  std::shared_ptr<IDTetris<int64_t>> t2(new IDTetris<int64_t>("t2"));
  std::shared_ptr<FIFOTetris<int64_t>> t1(new FIFOTetris<int64_t>("t1"));

  t2->add(c4);
  t1->add(c3);
  t1->add(t2);
  tetris.add(c1);
  tetris.add(c2);
  tetris.add(t1);
  EXPECT_EQ(tetris.HasReady(), false);

  c1->Push(3, 4);
  c1->Push(4, 2);
  c1->Push(5, 6);
  c1->Push(1328, 8);
  c1->Push(1329, 9);
  EXPECT_EQ(tetris.HasReady(), false);

  c2->Push(13);
  c2->Push(14, 2);
  c2->Push(15, 5);
  c2->Push(16, 9);
  c2->Push(17, 8);
  EXPECT_EQ(tetris.HasReady(), false);

  c3->Push(11);
  c3->Push(114, 2);
  c3->Push(115, 5);
  c3->Push(117, 8);
  EXPECT_EQ(tetris.HasReady(), false);

  c4->Push(1150, 5);
  c4->Push(1170, 8);
  c4->Push(110);
  EXPECT_EQ(tetris.HasReady(), true);

  auto rs = tetris.Pop();
  EXPECT_EQ(rs["c1"].first, std::make_pair(4, 3).first);
  EXPECT_EQ(rs["c1"].second, std::make_pair(4, 3).second);

  EXPECT_EQ(rs["c2"].first, std::make_pair(-1, 13).first);
  EXPECT_EQ(rs["c2"].second, std::make_pair(-1, 13).second);

  EXPECT_EQ(rs["c3"].first, std::make_pair(-1, 11).first);
  EXPECT_EQ(rs["c3"].second, std::make_pair(-1, 11).second);

  EXPECT_EQ(rs["c4"].first, std::make_pair(-1, 110).first);
  EXPECT_EQ(rs["c4"].second, std::make_pair(-1, 110).second);

  EXPECT_EQ(tetris.HasReady(), true);
  rs = tetris.Pop();
  EXPECT_EQ(tetris.HasReady(), true);
  rs = tetris.Pop();
  EXPECT_EQ(tetris.HasReady(), false);
}

}  // namespace caffe
