/*
 * grpc_server_init_test.cpp
 * Copyright (C) 2017 xiaoshu <2012wxs@gmail.com>
 *
 * Distributed under terms of the MIT license.
 */

#include "distributed_runtime/grpc_server_init.h"

#include "distributed_runtime/topology.pb.h"
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

DEFINE_string(topology, "/home/xiaoshu/dl_sys/oneflow/oneflow/distributed_runtime/topology.txt", "");

namespace oneflow {

TEST(GrpcServer, test) {
  oneflow::Topology topology;
  //std::shared_ptr<GrpcServer> grpc_sever_(new GrpcServer());
  GrpcServer* grpc_sever_(new GrpcServer());
  grpc_sever_->InitTopology(topology, FLAGS_topology);
  for(auto& v : grpc_sever_->vec_){
    std::cout<<v<<std::endl;
  }
  //EXPECT_EQ(grpc_sever_->vec_[0], "1");
  //EXPECT_EQ(grpc_sever_->vec_[1], "2");
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

}



