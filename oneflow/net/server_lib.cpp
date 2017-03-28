/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include <iostream>
#include "net/server_lib.h"

#include <unordered_map>


namespace oneflow {

namespace {
typedef std::unordered_map<std::string, ServerFactory*> ServerFactories;
ServerFactories* server_factories() {
  static ServerFactories* factories = new ServerFactories;
  return factories;
}
}  // namespace

/* static */
void ServerFactory::Register(const std::string& server_type,
                             ServerFactory* factory) {
  server_factories()->insert({server_type, factory}).second; 
}

/* static */
void ServerFactory::GetFactory(ServerFactory** out_factory) {
  // TODO(mrry): Improve the error reporting here.
  for (const auto& server_factory : *server_factories()) {
      *out_factory = server_factory.second;
  }
}

// Creates a server based on the given `server_def`, and stores it in
// `*out_server`. Returns OK on success, otherwise returns an error.
void NewServer(const ServerDef& server_def, std::unique_ptr<ServerInterface>* out_server) {
  std::cout<<"new server in server_lib.cc"<<std::endl;
  ServerFactory* factory;
  ServerFactory::GetFactory(&factory);
  std::cout<<"get factory in server_lib.cc"<<std::endl;
  factory->NewServer(server_def, out_server);
  std::cout<<"hi"<<std::endl;
}

}  // namespace tensorflow
