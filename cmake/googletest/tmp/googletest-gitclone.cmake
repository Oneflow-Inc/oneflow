if("ec44c6c1675c25b9827aacd08c02433cccde7780" STREQUAL "")
  message(FATAL_ERROR "Tag for git checkout should not be empty.")
endif()

set(run 0)

if("/home/xiaoshu/dl_sys/oneflow/cmake/googletest/src/googletest-stamp/googletest-gitinfo.txt" IS_NEWER_THAN "/home/xiaoshu/dl_sys/oneflow/cmake/googletest/src/googletest-stamp/googletest-gitclone-lastrun.txt")
  set(run 1)
endif()

if(NOT run)
  message(STATUS "Avoiding repeated git clone, stamp file is up to date: '/home/xiaoshu/dl_sys/oneflow/cmake/googletest/src/googletest-stamp/googletest-gitclone-lastrun.txt'")
  return()
endif()

execute_process(
  COMMAND ${CMAKE_COMMAND} -E remove_directory "/home/xiaoshu/dl_sys/oneflow/cmake/googletest/src/googletest"
  RESULT_VARIABLE error_code
  )
if(error_code)
  message(FATAL_ERROR "Failed to remove directory: '/home/xiaoshu/dl_sys/oneflow/cmake/googletest/src/googletest'")
endif()

set(git_options)

# disable cert checking if explicitly told not to do it
set(tls_verify "")
if(NOT "x" STREQUAL "x" AND NOT tls_verify)
  list(APPEND git_options
    -c http.sslVerify=false)
endif()

set(git_clone_options)

set(git_shallow "")
if(git_shallow)
  list(APPEND git_clone_options --depth 1 --no-single-branch)
endif()

# try the clone 3 times incase there is an odd git clone issue
set(error_code 1)
set(number_of_tries 0)
while(error_code AND number_of_tries LESS 3)
  execute_process(
    COMMAND "/usr/bin/git" ${git_options} clone ${git_clone_options} --origin "origin" "https://github.com/google/googletest.git" "googletest"
    WORKING_DIRECTORY "/home/xiaoshu/dl_sys/oneflow/cmake/googletest/src"
    RESULT_VARIABLE error_code
    )
  math(EXPR number_of_tries "${number_of_tries} + 1")
endwhile()
if(number_of_tries GREATER 1)
  message(STATUS "Had to git clone more than once:
          ${number_of_tries} times.")
endif()
if(error_code)
  message(FATAL_ERROR "Failed to clone repository: 'https://github.com/google/googletest.git'")
endif()

execute_process(
  COMMAND "/usr/bin/git" ${git_options} checkout ec44c6c1675c25b9827aacd08c02433cccde7780
  WORKING_DIRECTORY "/home/xiaoshu/dl_sys/oneflow/cmake/googletest/src/googletest"
  RESULT_VARIABLE error_code
  )
if(error_code)
  message(FATAL_ERROR "Failed to checkout tag: 'ec44c6c1675c25b9827aacd08c02433cccde7780'")
endif()

execute_process(
  COMMAND "/usr/bin/git" ${git_options} submodule init 
  WORKING_DIRECTORY "/home/xiaoshu/dl_sys/oneflow/cmake/googletest/src/googletest"
  RESULT_VARIABLE error_code
  )
if(error_code)
  message(FATAL_ERROR "Failed to init submodules in: '/home/xiaoshu/dl_sys/oneflow/cmake/googletest/src/googletest'")
endif()

execute_process(
  COMMAND "/usr/bin/git" ${git_options} submodule update --recursive --init 
  WORKING_DIRECTORY "/home/xiaoshu/dl_sys/oneflow/cmake/googletest/src/googletest"
  RESULT_VARIABLE error_code
  )
if(error_code)
  message(FATAL_ERROR "Failed to update submodules in: '/home/xiaoshu/dl_sys/oneflow/cmake/googletest/src/googletest'")
endif()

# Complete success, update the script-last-run stamp file:
#
execute_process(
  COMMAND ${CMAKE_COMMAND} -E copy
    "/home/xiaoshu/dl_sys/oneflow/cmake/googletest/src/googletest-stamp/googletest-gitinfo.txt"
    "/home/xiaoshu/dl_sys/oneflow/cmake/googletest/src/googletest-stamp/googletest-gitclone-lastrun.txt"
  WORKING_DIRECTORY "/home/xiaoshu/dl_sys/oneflow/cmake/googletest/src/googletest"
  RESULT_VARIABLE error_code
  )
if(error_code)
  message(FATAL_ERROR "Failed to copy script-last-run stamp file: '/home/xiaoshu/dl_sys/oneflow/cmake/googletest/src/googletest-stamp/googletest-gitclone-lastrun.txt'")
endif()

