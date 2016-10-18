cmake_minimum_required(VERSION 2.8)
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Build type")
    message(STATUS "No build type selected, default to ${CMAKE_BUILD_TYPE}")
endif()
project(block_spmv)

#----------------------------------------------------------------------------
# Find OpenMP
#----------------------------------------------------------------------------
find_package(OpenMP)
if (OpenMP_CXX_FLAGS)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif ()


#----------------------------------------------------------------------------
# Find Boost
#----------------------------------------------------------------------------
set(BOOST_COMPONENTS ${BOOST_COMPONENTS}
    program_options
    system
    filesystem
    )

find_package(Boost COMPONENTS ${BOOST_COMPONENTS})
include_directories( ${Boost_INCLUDE_DIRS} )

#----------------------------------------------------------------------------
# Find VexCL
#----------------------------------------------------------------------------
set(VEXCL_ROOT $ENV{VEXCL_ROOT} CACHE STRING "VexCL root")
if (VEXCL_ROOT)
    include_directories( ${VEXCL_ROOT} )

    set(VEXCL_BACKEND "OpenCL" CACHE STRING "Select VexCL backend (OpenCL/CUDA)")
    set_property(CACHE VEXCL_BACKEND PROPERTY STRINGS "OpenCL" "CUDA")

    #----------------------------------------------------------------------------
    # Find VexCL backend
    #----------------------------------------------------------------------------
    if ("${VEXCL_BACKEND}" STREQUAL "OpenCL")
        find_package(OpenCL REQUIRED)
        include_directories( ${OPENCL_INCLUDE_DIRS} )
        set(VEXCL_BACKEND_LIBRARIES ${OpenCL_LIBRARIES})
        add_definitions(-DVEXCL_BACKEND_OPENCL)
    elseif ("${VEXCL_BACKEND}" STREQUAL "CUDA")
        find_package(CUDA REQUIRED)
        include_directories( ${CUDA_INCLUDE_DIRS} )
        set(VEXCL_BACKEND_LIBRARIES ${CUDA_CUDA_LIBRARY})
        add_definitions(-DVEXCL_BACKEND_CUDA)
    endif()
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-ignored-attributes -Wno-deprecated-declarations")

#----------------------------------------------------------------------------
# Find AMGCL
#----------------------------------------------------------------------------
find_path(AMGCL_INCLUDE amgcl/amg.h)
if (AMGCL_INCLUDE)
    include_directories(${AMGCL_INCLUDE})
endif()


add_executable(bench bench.cpp)
target_link_libraries(bench ${Boost_LIBRARIES} ${VEXCL_BACKEND_LIBRARIES})

add_executable(bench_BB bench_BB.cpp)
target_link_libraries(bench_BB ${Boost_LIBRARIES} ${VEXCL_BACKEND_LIBRARIES})


