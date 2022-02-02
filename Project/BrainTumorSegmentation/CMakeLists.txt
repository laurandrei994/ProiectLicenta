project (BrainTumorSegmentation)

SET(CMAKE_AUTOUIC ON)
SET(CMAKE_AUTOMOC ON)
SET(CMAKE_INCLUDE_CURRENT_DIR ON)

FIND_PACKAGE (Qt5Widgets REQUIRED)

add_subdirectory("include")
add_subdirectory("src")

source_group("Header Files" FILES ${headers})
source_group("Source Files" FILES ${sources})

include_directories(${Algorithms_IncludeDir} ${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIB_DIR})

qt5_wrap_ui(UIS_HDRS ${CMAKE_CURRENT_SOURCE_DIR}/src/brainTumor.ui)

add_executable (BrainTumorSegmentation ${sources} ${headers} ${UIS_HDRS})

SET (Qt5_DLL_Path "C:/Qt/5.15.0/msvc2019_64/bin")
message("Qt5 Part")
message(STATUS "	version: ${Qt5Widgets_VERSION}")
message(STATUS "	libraries: ${Qt5Widgets_LIBRARIES}")
message(STATUS "	Qt5_pathDebug: ${Qt5_DLL_Path}")
message(STATUS "	Qt5_pathRelease: ${Qt5_DLL_Path}")
message("	CMAKE_BINARY_DIR: ${CMAKE_BINARY_DIR}")

SET (Qt5_DLLs_D ${Qt5_DLL_Path}/Qt5Widgetsd.dll
				${Qt5_DLL_Path}/Qt5Cored.dll
				${Qt5_DLL_Path}/Qt5Guid.dll	)
SET (QT5_PDBs_D ${Qt5_DLL_Path}/Qt5Widgetsd.pdb
				${Qt5_DLL_Path}/Qt5Cored.pdb
				${Qt5_DLL_Path}/Qt5Guid.pdb	)
SET (Qt5_DLLs_R ${Qt5_DLL_Path}/Qt5Widgets.dll
				${Qt5_DLL_Path}/Qt5Core.dll
				${Qt5_DLL_Path}/Qt5Gui.dll	)
SET (Qt5_PDBs_R ${Qt5_DLL_Path}/Qt5Widgets.pdb
				${Qt5_DLL_Path}/Qt5Core.pdb
				${Qt5_DLL_Path}/Qt5Gui.pdb	)

install (FILES ${Qt5_DLLs_D} DESTINATION ${CMAKE_BINARY_DIR}/build/bin/X64/Debug CONFIGURATIONS Debug)
install (FILES ${Qt5_PDBs_D} DESTINATION ${CMAKE_BINARY_DIR}/build/bin/X64/Debug CONFIGURATIONS Debug)
install (FILES ${Qt5_DLLs_R} DESTINATION ${CMAKE_BINARY_DIR}/build/bin/X64/Release CONFIGURATIONS Release)
install (FILES ${Qt5_PDBs_R} DESTINATION ${CMAKE_BINARY_DIR}/build/bin/X64/Release CONFIGURATIONS Release)

target_link_libraries (BrainTumorSegmentation Algorithms Qt5::Widgets ${OpenCV_LIBS})

if (WIN32)
	if($<CONFIG:Debug>)
		get_target_property(WIDGETDLL Qt5::Widgets IMPORTED_LOCATION_DEBUG)
	else($<CONFIG:Debug>)
		get_target_property(WIDGETDLL Qt5::Widgets IMPORTED_LOCATION_RELEASE)
	endif($<CONFIG:Debug>)
	
	add_custom_command(TARGET BrainTumorSegmentation POST_BUILD
						COMMAND ${CMAKE_COMMAND} -E copy
						$<TARGET_FILE:Qt5::Widgets>
						$<TARGET_FILE:Qt5::Core>
						$<TARGET_FILE:Qt5::Gui>
						$<TARGET_FILE_DIR:BrainTumorSegmentation>)
endif(WIN32)