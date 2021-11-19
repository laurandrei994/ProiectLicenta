cmake_minimum_required (VERSION 3.8)
project (DenoisingTest)

set(OpenCV_STATIC OFF)
set(DenoisingTest_IncludeDir "${CMAKE_CURRENT_SOURCE_DIR}/include" CACHE INTERNAL "")

FIND_PACKAGE(OpenCV REQUIRED)
if (NOT OpenCV_FOUND)
	message( FATAL_ERROR "Cannot build without OpenCV! Please provide location of OpenCV make files!")
endif()

add_subdirectory("src")
add_subdirectory("include")

source_group ("Header Files" FILES ${headers})
source_group ("Source Files" FILES ${sources})

include_directories (${DenoisingTest_IncludeDir})
link_directories (${OpenCV_LIB_DIR})

add_executable (DenoisingTest ${sources} ${headers})

message("OpenCV library status:")
message(STATUS "	version: ${OpenCV_VERSION}")
message(STATUS "	libraries: ${OpenCV_LIBS}")
message(STATUS "	include_path: ${OpenCV_INCLUDE_DIRS}")

if(CMAKE_CL_64)
	set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/build/bin/X64)
	SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/build/lib/X64)
	SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/build/lib/X64)
else(CMAKE_CL_64)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/build/bin/X86)
	SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/build/lib/X86)
	SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/build/lib/X86)
endif(CMAKE_CL_64)

set(OpenCV_DLL_Path "${OpenCV_DIR}/x64/vc16/bin")
message("OpenCV Paths:")
message(STATUS "	OpenCV_DIR: ${OpenCV_DIR}")
message(STATUS "	Debug_Path: ${OpenCV_DLL_Path}")
message(STATUS "	Release_Path: ${OpenCV_DLL_Path}")
message("CMake_BINARY_DIR: ${CMAKE_BINARY_DIR}")

SET (OpenCV_DLLs_D 	${OpenCV_DLL_Path}/opencv_core452d.dll
					${OpenCV_DLL_Path}/opencv_highgui452d.dll
					${OpenCV_DLL_Path}/opencv_imgcodecs452d.dll
					${OpenCV_DLL_Path}/opencv_imgproc452d.dll
					${OpenCV_DLL_Path}/opencv_videoio452d.dll	)

SET (OpenCV_DLLs_R 	${OpenCV_DLL_Path}/opencv_core452.dll
					${OpenCV_DLL_Path}/opencv_highgui452.dll
					${OpenCV_DLL_Path}/opencv_imgcodecs452.dll
					${OpenCV_DLL_Path}/opencv_imgproc452.dll
					${OpenCV_DLL_Path}/opencv_videoio452.dll	)
				
install (FILES ${OpenCV_DLLs_D} DESTINATION ${CMAKE_BINARY_DIR}/Debug CONFIGURATIONS Debug)
install (FILES ${OpenCV_DLLs_R} DESTINATION ${CMAKE_BINARY_DIR}/Release CONFIGURATIONS Release)

target_link_libraries (DenoisingTest ${OpenCV_LIBS})