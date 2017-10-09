CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

PROJECT(sceneCreator)

set (CMAKE_BUILD_TYPE Release)
SET(SRC
src/createscene.cpp 
src/tinyply.h
src/tinyply.cpp
)
SET(loc
src/estimate_local_sym.cpp
src/tinyply.h
src/tinyply.cpp
)
SET(glob
src/estimate_global_sym.cpp
src/tinyply.h
src/tinyply.cpp
)

FIND_PACKAGE( OpenCV REQUIRED )

find_package(Boost 1.40 REQUIRED COMPONENTS filesystem program_options system thread)
find_package(PkgConfig)
pkg_search_module(Eigen3 REQUIRED eigen3)

add_executable(createScene ${SRC})
target_link_libraries(createScene  ${Boost_LIBRARIES})


add_executable(estimate_local_sym ${loc})
target_link_libraries(estimate_local_sym ${OpenCV_LIBS} ${Boost_LIBRARIES})

add_executable(estimate_global_sym ${glob})
target_link_libraries(estimate_global_sym ${OpenCV_LIBS} ${Boost_LIBRARIES})
set(CMAKE_CXX_FLAGS "-w -fpermissive -std=c++11")





