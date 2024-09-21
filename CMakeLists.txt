project(gtsam_quadrics)
cmake_minimum_required(VERSION 3.18)
set(GTSAM_QUADRICS_VERSION 0.1.2)

# we want to use some C++11 features
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wextra")

###################################################################################
find_package(GTSAM REQUIRED)
find_package(GTSAMCMakeTools REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(${EIGEN_INCLUDE_DIR})
include_directories(${GTSAM_INCLUDE_DIR})

include(GtsamBuildTypes) 			# Load build type flags and default to Debug mode
include(GtsamTesting)    			# Easy functions for creating unit tests and scripts
include(GtsamMakeConfigFile) 

###################################################################################
# add gtsam_quadrics

# include headers, ensure that local folder is searched before library folders
include_directories(BEFORE "${PROJECT_SOURCE_DIR}")

# set source files
set(SOURCE_FILES
  ./gtsam_quadrics/base/Utilities.cpp
  ./gtsam_quadrics/geometry/ConstrainedDualQuadric.cpp
  ./gtsam_quadrics/geometry/AlignedBox2.cpp
  ./gtsam_quadrics/geometry/AlignedBox3.cpp
  ./gtsam_quadrics/geometry/BoundingBoxFactor.cpp
  ./gtsam_quadrics/geometry/QuadricAngleFactor.cpp
  ./gtsam_quadrics/geometry/QuadricCamera.cpp
  ./gtsam_quadrics/geometry/DualConic.cpp
)

###################################################################################
## Build static library from common sources
set(CONVENIENCE_LIB_NAME "gtsam_quadrics")
add_library(${CONVENIENCE_LIB_NAME} ${SOURCE_FILES})
target_link_libraries(${CONVENIENCE_LIB_NAME} gtsam)

###################################################################################
# install library and PACKAGEConfig.cmake
install(
  TARGETS ${CONVENIENCE_LIB_NAME}
  EXPORT GTSAM_QUADRICS-exports
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin)
GtsamMakeConfigFile(GTSAM_QUADRICS)
# list(APPEND GTSAM_EXPORTED_TARGETS ${CONVENIENCE_LIB_NAME})
# export(TARGETS ${GTSAM_EXPORTED_TARGETS} FILE GTSAM-exports.cmake)

file(GLOB gtsam_quadrics_base_headers "gtsam_quadrics/base/*.h")
install(FILES ${gtsam_quadrics_base_headers} DESTINATION include/gtsam_quadrics/base)
file(GLOB gtsam_quadrics_geometry_headers "gtsam_quadrics/geometry/*.h")
install(FILES ${gtsam_quadrics_geometry_headers} DESTINATION include/gtsam_quadrics/geometry)

###################################################################################
# Build example files (CMake tracks the dependecy to link with GTSAM through our project's static library)
# TODO fix broken examples!
# gtsamAddExamplesGlob("*.cpp" "" "${CONVENIENCE_LIB_NAME}")
# build examples if make examples
# https://stackoverflow.com/questions/17765618/cmake-add-subdirectories-to-custom-target-and-adding-custom-target-to-vs-solut
# TODO: pass ${CONVENIENCE_LIB_NAME} to examples/c++/CMakeLists.txt
# add_custom_target(examples) <-- already exists from GtsamTesting
# add_subdirectory(examples/c++)
# add_dependencies(examples examples_folder)

###################################################################################
# Build Doxygen documentation

# find doxygen
find_package(Doxygen)

# add subdirectory with custom make doc, doc_clean targets
if (DOXYGEN_FOUND)
  add_subdirectory(doc)
endif()
