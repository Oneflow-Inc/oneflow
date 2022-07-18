#ifndef ONEFLOW_API_CPP_ONE_EMBEDDING_ONE_EMBEDDING_H_
#define ONEFLOW_API_CPP_ONE_EMBEDDING_ONE_EMBEDDING_H_

#include "../../framework.h"

namespace oneflow_api {
namespace one_embedding {

void LoadSnapshot(const std::string& embedding_name, const std::string& snapshot_name);

}

}  // namespace oneflow_api

#endif  // ONEFLOW_API_CPP_ONE_EMBEDDING_ONE_EMBEDDING_H_
