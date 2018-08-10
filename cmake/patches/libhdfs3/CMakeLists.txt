CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

PROJECT(libhdfs3)

SET(CMAKE_VERBOSE_MAKEFILE ON CACHE STRING "Verbose build." FORCE)

IF(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
    MESSAGE(FATAL_ERROR "cannot build the project in the source directory! Out-of-source build is enforced!")
ENDIF()

SET(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CMake" ${CMAKE_MODULE_PATH})
SET(DOXYFILE_PATH ${CMAKE_SOURCE_DIR}/docs)

INCLUDE(Platform)
INCLUDE(Functions)
INCLUDE(Options)

MESSAGE(STATUS "=========> in patch")

FIND_PACKAGE(LibXml2 REQUIRED)
FIND_PACKAGE(Protobuf REQUIRED)
FIND_PACKAGE(KERBEROS REQUIRED)
FIND_PACKAGE(GSasl REQUIRED)
FIND_PACKAGE(GoogleTest REQUIRED)
FIND_PACKAGE(SSL REQUIRED)
FIND_PACKAGE(CURL REQUIRED)
INCLUDE_DIRECTORIES(${GoogleTest_INCLUDE_DIR})
LINK_LIBRARIES(${GoogleTest_LIBRARIES})

IF(OS_LINUX)
    FIND_PACKAGE(LibUUID REQUIRED)
ENDIF(OS_LINUX)

ADD_SUBDIRECTORY(mock)
ADD_SUBDIRECTORY(src)
ADD_SUBDIRECTORY(test)

CONFIGURE_FILE(src/libhdfs3.pc.in ${CMAKE_SOURCE_DIR}/src/libhdfs3.pc @ONLY)
CONFIGURE_FILE(debian/changelog.in ${CMAKE_SOURCE_DIR}/debian/changelog @ONLY)

ADD_CUSTOM_TARGET(debian-package
	COMMAND dpkg-buildpackage -us -uc -b
	WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
	COMMENT "Create debian package..."
)

ADD_CUSTOM_TARGET(rpm-package
	COMMAND rpmbuild -bb --define "_topdir ${CMAKE_SOURCE_DIR}/rpms" --define "version ${libhdfs3_VERSION_STRING}" ${CMAKE_SOURCE_DIR}/rpms/libhdfs3.spec
	WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
	COMMENT "Create rpm package..."
)

ADD_CUSTOM_TARGET(doc
	COMMAND doxygen ${CMAKE_BINARY_DIR}/src/doxyfile
	WORKING_DIRECTORY ${DOXYFILE_PATH}
	COMMENT "Generate documents..."
)

ADD_CUSTOM_TARGET(style
	COMMAND astyle --style=attach --indent=spaces=4 --indent-preprocessor --break-blocks --pad-oper --pad-header --unpad-paren --delete-empty-lines --suffix=none --align-pointer=middle --lineend=linux --indent-col1-comments ${libhdfs3_SOURCES}
	COMMAND astyle --style=attach --indent=spaces=4 --indent-preprocessor --break-blocks --pad-oper --pad-header --unpad-paren --delete-empty-lines --suffix=none --align-pointer=middle --lineend=linux --indent-col1-comments ${unit_SOURCES}
	COMMAND astyle --style=attach --indent=spaces=4 --indent-preprocessor --break-blocks --pad-oper --pad-header --unpad-paren --delete-empty-lines --suffix=none --align-pointer=middle --lineend=linux --indent-col1-comments ${function_SOURCES}
	COMMAND astyle --style=attach --indent=spaces=4 --indent-preprocessor --break-blocks --pad-oper --pad-header --unpad-paren --delete-empty-lines --suffix=none --align-pointer=middle --lineend=linux --indent-col1-comments ${secure_SOURCES}
	WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
	COMMENT "format code style..."
)
