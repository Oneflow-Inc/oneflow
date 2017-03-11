#include <glog/logging.h>
#include <gflags/gflags.h>
#include <iostream>
#include <string>
#include "caffe.pb.h"
#include "proto_io.h"
#include "context/one.h"
#include "context/machine_descriptor.h"
#include "context/id_map.h"
#include "memory/blob.h"

DEFINE_string(solver, "",
  "The solver definition protocol buffer text file.");

int main(int argc, char* argv[]) {

  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = 1;
  DLOG(INFO) << "JOB TOOL";

  // Change Dtype if another type is needed
  // For example, using Dtype = double
  using Dtype = float;
  caffe::SolverProto solver_param;
  caffe::ReadProtoFromTextFileOrDie(FLAGS_solver, &solver_param);
  caffe::TheOne<Dtype>::InitResource(FLAGS_solver);
  caffe::TheOne<Dtype>::InitJob2(solver_param);
  // caffe::TheOne<Dtype>::InitThread();
  // caffe::TheOne<Dtype>::FinalizeThread();
  

  DLOG(INFO) << "JOB TOOL END";
  return 0;
}
