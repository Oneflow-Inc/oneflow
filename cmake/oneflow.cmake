# main cpp
list(APPEND of_main_cc ${oneflow_src_dir}/compiler/compiler.cpp)
list(APPEND of_main_cc ${oneflow_src_dir}/runtime/elf_runner.cpp)

# source_group
SUBDIRLIST(subdir_list ${oneflow_src_dir})
foreach(subdir ${subdir_list})
  file(GLOB subdir_headers    "${oneflow_src_dir}/${subdir}/*.h")
  file(GLOB subdir_cuda_headers    "${oneflow_src_dir}/${subdir}/*.cuh")
  file(GLOB subdir_obj_cpps   "${oneflow_src_dir}/${subdir}/*.cpp")
  file(GLOB subdir_obj_cus   "${oneflow_src_dir}/${subdir}/*.cu")
  file(GLOB subdir_test_cpps  "${oneflow_src_dir}/${subdir}/*_test.cpp")
  file(GLOB subdir_protos     "${oneflow_src_dir}/${subdir}/*.proto")
  foreach(test_cpp ${subdir_test_cpps})
    list(REMOVE_ITEM subdir_obj_cpps ${test_cpp})
  endforeach()
  foreach(main_cpp ${of_main_cc})
    list(REMOVE_ITEM subdir_obj_cpps ${main_cpp})
  endforeach()
  source_group(${subdir} FILES ${subdir_headers} ${subdir_cuda_headers} ${subdir_obj_cpps} ${subdir_obj_cus} {subdir_protos})
  list(APPEND of_all_obj_cc ${subdir_headers} ${subdir_cuda_headers} ${subdir_obj_cpps} ${subdir_obj_cus})
  list(APPEND of_all_proto ${subdir_protos})
  list(APPEND of_all_test_cc ${subdir_test_cpps})
endforeach()

# proto obj lib
foreach(proto_name ${of_all_proto})
  file(RELATIVE_PATH proto_rel_name ${oneflow_src_dir} ${proto_name})
  list(APPEND of_all_rel_protos ${proto_rel_name})
endforeach()

RELATIVE_PROTOBUF_GENERATE_CPP(PROTO_SRCS PROTO_HDRS
                               ${oneflow_src_dir}
                               ${of_all_rel_protos})

cuda_add_library(of_protoobj ${PROTO_SRCS} ${PROTO_HDRS})
target_link_libraries(of_protoobj ${oneflow_third_party_libs})

# cc obj lib
include_directories(${oneflow_src_dir})
include_directories(${PROJECT_BINARY_DIR})
cuda_add_library(of_ccobj ${of_all_obj_cc})
target_link_libraries(of_ccobj ${oneflow_third_party_libs})
add_dependencies(of_ccobj of_protoobj)

if(APPLE)
  set(of_libs -Wl,-force_load of_ccobj of_protoobj)
elseif(UNIX)
  set(of_libs -Wl,--whole-archive of_ccobj of_protoobj -Wl,--no-whole-archive)
elseif(WIN32)
  set(of_libs of_ccobj of_protoobj)
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /WHOLEARCHIVE:of_ccobj") 
endif()

# build main
foreach(cc ${of_main_cc})
  get_filename_component(main_name ${cc} NAME_WE)
  cuda_add_executable(${main_name} ${cc})
  target_link_libraries(${main_name} ${of_libs} ${oneflow_third_party_libs})
endforeach()

# build test
foreach(cc ${of_all_test_cc})
  get_filename_component(test_name ${cc} NAME_WE)
  string(CONCAT test_exe_name ${test_name} exe)
  cuda_add_executable(${test_exe_name} ${cc})
  target_link_libraries(${test_exe_name} ${of_libs} ${oneflow_third_party_libs})
  add_test(NAME ${test_name} COMMAND ${test_exe_name})
endforeach()
