cmake_minimum_required(VERSION 2.8)
project(Line_sign)

set( CMAKE_BUILD_TYPE "Debug" )
set( CMAKE_CXX_COMPLIER "g++" )
set( CMAKE_CXX_FLAGS "-std=c++11 -march=native -O3")

set( EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin )
set( LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib )

list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )

find_package( OpenCV 3.4 REQUIRED )

include_directories( ${OpenCV_INCLUDE_DIRS}
			${PROJECT_SOURCE_DIR}/include
)

set( THIRD_PARTITY_LIBS
	${OpenCV_LIBRARIES}
)

add_subdirectory( sq )
