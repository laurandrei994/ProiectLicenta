cmake_minimum_required (VERSION 3.8)

project (Segmentation)

set(CMAKE_CXX_STANDARD_REQUIRED ON)

set (OpenCV_STATIC OFF)

FIND_PACKAGE(OpenCV REQUIRED)
if( NOT OpenCV_FOUND )
    message( FATAL_ERROR "Cannot build without OpenCV! Please provide location of OpenCV cmake files!" )
endif()

message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

if(CMAKE_CL_64)
	set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/build/bin/X64)
	SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/build/lib/X64)
	SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/build/lib/X64)
else(CMAKE_CL_64)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/build/bin/X86)
	SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/build/lib/X86)
	SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/build/lib/X86)
endif(CMAKE_CL_64)

add_subdirectory (Algorithms)
add_subdirectory (BrainTumorSegmentation)

add_dependencies (BrainTumorSegmentation Algorithms)