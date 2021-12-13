// #include "oneflow/core/common/util.h"
// #include "oneflow/core/ep/cpu/parallel.h"

// #if WITH_OMP_THREADING_RUNTIME

// size_t get_computing_cores()
// {
//   auto envar = std::getenv("ONEFLOW_CPU_CORES");
//   if (envar) {
//       std::cout<< "ONEFLOW_CPU_CORES : " << envar << std::endl;
//   }

//   // return omp_get_max_threads();
//   return 4;
// }


// COMMAND(omp_set_num_threads(get_computing_cores()););

// #endif