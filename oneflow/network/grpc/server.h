/*
 * server.cpp
 * Copyright (C) 2017 wangxiaoshu <2012wxs@gmail.com>
 *
 * Distributed under terms of the MIT license.
 */

#include "network/network.h"

namespace oneflow {

struct NetworkMessage;
struct MemoryDescriptor;
struct NetworkMemory;

class Server : Network {
  public:
    Server();
    ~Server();
    bool Send(const NetworkMessage& msg);
    void Read(MemoryDescriptor* src, NetworkMemory* dst);
};

}



