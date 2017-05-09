/*
 * client.cpp
 * Copyright (C) 2017 xiaoshu <2012wxs@gmail.com>
 *
 * Distributed under terms of the MIT license.
 */

#include "network/grpc/client.h"

namespace oneflow {

Client::Client() {}
Client::~Client() {}


bool Client::Send(const NetworkMessage& msg) {
  return true;
}

void Read(MemoryDescriptor* src, NetworkMemory* dst) {

}

}



