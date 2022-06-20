message("-- LLVM_MONO_REPO_URL: " ${LLVM_MONO_REPO_URL})
message("-- LLVM_MONO_REPO_MD5: " ${LLVM_MONO_REPO_MD5})
FetchContent_Declare(llvm_monorepo)
FetchContent_GetProperties(llvm_monorepo)

if(NOT llvm_monorepo_POPULATED)
  FetchContent_Populate(llvm_monorepo URL ${LLVM_MONO_REPO_URL} URL_HASH MD5=${LLVM_MONO_REPO_MD5})
  set(LLVM_INSTALL_DIR ${THIRD_PARTY_DIR}/llvm)

  execute_process(
    COMMAND
      "${CMAKE_COMMAND}" ${llvm_monorepo_SOURCE_DIR}/llvm
      -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} # this is required in newer version of LLVM
      -DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER}
      -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}
      -DCMAKE_CUDA_COMPILER_LAUNCHER=${CMAKE_CUDA_COMPILER_LAUNCHER}
      -DCMAKE_EXE_LINKER_FLAGS_INIT=${CMAKE_EXE_LINKER_FLAGS_INIT}
      -DCMAKE_MODULE_LINKER_FLAGS_INIT=${CMAKE_MODULE_LINKER_FLAGS_INIT}
      -DCMAKE_SHARED_LINKER_FLAGS_INIT=${CMAKE_SHARED_LINKER_FLAGS_INIT}
      -DCMAKE_INSTALL_PREFIX=${LLVM_INSTALL_DIR} -DCMAKE_INSTALL_MESSAGE=${CMAKE_INSTALL_MESSAGE}
      -DLLVM_ENABLE_RTTI=ON # turn this on to make it compatible with protobuf
      -DLLVM_ENABLE_EH=ON # turn this on to make it compatible with half (the library)
      -DLLVM_BUILD_EXAMPLES=OFF -DLLVM_BUILD_TOOLS=OFF -DLLVM_INCLUDE_EXAMPLES=OFF
      -DLLVM_INCLUDE_TESTS=OFF -DLLVM_INCLUDE_BENCHMARKS=OFF -DLLVM_TARGETS_TO_BUILD=host\;NVPTX
      -DLLVM_ENABLE_ASSERTIONS=ON -DLLVM_ENABLE_PROJECTS=mlir -DLLVM_APPEND_VC_REV=OFF
      -DLLVM_ENABLE_ZLIB=OFF -DLLVM_INSTALL_UTILS=ON -DBUILD_SHARED_LIBS=${BUILD_SHARED_LIBS}
      -DLLVM_ENABLE_OCAMLDOC=OFF -DLLVM_ENABLE_BINDINGS=OFF
      -DLLVM_ENABLE_TERMINFO=OFF # Disable terminfo in llvm so that oneflow doesn't need to link against it
      -DMLIR_ENABLE_CUDA_RUNNER=${WITH_MLIR_CUDA_CODEGEN}
      -DCMAKE_CUDA_COMPILER=${CMAKE_CUDA_COMPILER} -DINJA_URL=${INJA_URL}
      -DINJA_URL_HASH=${INJA_URL_HASH} -DJSON_URL=${JSON_URL} -DJSON_URL_HASH=${JSON_URL_HASH}
      -DCMAKE_CUDA_COMPILER=${CMAKE_CUDA_COMPILER} -DLLVM_EXTERNAL_PROJECTS=OneFlowTableGen
      -DLLVM_EXTERNAL_ONEFLOWTABLEGEN_SOURCE_DIR=${CMAKE_SOURCE_DIR}/tools/oneflow-tblgen -G
      ${CMAKE_GENERATOR}
    WORKING_DIRECTORY ${llvm_monorepo_BINARY_DIR}
    RESULT_VARIABLE ret)
  if(ret EQUAL "1")
    message(FATAL_ERROR "Bad exit status")
  endif()
  include(ProcessorCount)
  ProcessorCount(PROC_NUM)
  if(WITH_MLIR)
    set(INSTALL_ALL "install")
  endif()
  execute_process(
    COMMAND "${CMAKE_COMMAND}" --build . -j${PROC_NUM} --target ${INSTALL_ALL}
            install-oneflow-tblgen install-mlir-headers
    WORKING_DIRECTORY ${llvm_monorepo_BINARY_DIR} RESULT_VARIABLE ret)
  if(ret EQUAL "1")
    message(FATAL_ERROR "Bad exit status")
  endif()
endif()

set(LLVM_INCLUDE_DIRS ${llvm_monorepo_SOURCE_DIR}/llvm/include;${llvm_monorepo_BINARY_DIR}/include)

if(WITH_MLIR)
  set(LLVM_DIR ${LLVM_INSTALL_DIR}/lib/cmake/llvm)
  set(MLIR_DIR ${LLVM_INSTALL_DIR}/lib/cmake/mlir)
  find_package(MLIR REQUIRED CONFIG)

  message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
  message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

  set(MLIR_BINARY_DIR ${llvm_monorepo_BINARY_DIR})

  list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
  include(TableGen)
  include(AddLLVM)
  include(AddMLIR)
  include(HandleLLVMOptions)
  set(LLVM_EXTERNAL_LIT "${llvm_monorepo_BINARY_DIR}/bin/llvm-lit" CACHE STRING "" FORCE)
endif()
