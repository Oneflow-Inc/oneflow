#ifndef ONEFLOW_CORE_THREAD_THREAD_UNIQUE_TAG_H_
#define ONEFLOW_CORE_THREAD_THREAD_UNIQUE_TAG_H_

#include <string>
#include "oneflow/core/common/maybe.h"

namespace oneflow {

Maybe<void> SetThisThreadUniqueTag(const std::string& thread_tag);
Maybe<const std::string&> GetThisThreadUniqueTag();

}

#endif  // ONEFLOW_CORE_THREAD_THREAD_UNIQUE_TAG_H_
