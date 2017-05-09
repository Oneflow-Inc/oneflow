/*
 * server.cpp
 * Copyright (C) 2017 xiaoshu <2012wxs@gmail.com>
 *
 * Distributed under terms of the MIT license.
 */

#include "network/grpc/server.h"

namespace oneflow {

Server::Server() {}
Server::~Server() {}

bool Server::Send(const NetworkMessage& msg) {
  return true;
}

void Server::Read(MemoryDescriptor* src, NetworkMemory* dst) {

}

}



