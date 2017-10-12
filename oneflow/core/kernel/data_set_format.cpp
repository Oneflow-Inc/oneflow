#include "oneflow/core/kernel/data_set_format.h"

namespace oneflow {

#define DATA_SET_OVERRITE_OFSTREAM(type)                          \
  std::ostream& operator<<(std::ostream& out, const type& data) { \
    out.write(reinterpret_cast<const char*>(&data), data.Size()); \
    return out;                                                   \
  }
OF_PP_FOR_EACH_TUPLE(DATA_SET_OVERRITE_OFSTREAM, DATA_SET_FORMAT_SEQ);
}
