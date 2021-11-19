#include "oneflow/core/ep/cpu/cpu_stream.h"

namespace oneflow {

namespace ep {


CpuStream::CpuStream()
{
  onednn_engine_.reset(new dnnl::engine(dnnl::engine::kind::cpu, 0));
  onednn_stream_.reset(new dnnl::stream(*onednn_engine_));
}



}

}