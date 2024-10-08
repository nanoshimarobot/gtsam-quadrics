find_package(GTSAM REQUIRED)
find_package(matplotlibcpp17 REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)

include_directories(BEFORE "${PROJECT_SOURCE_DIR}")
include_directories(${PYTHON_INCLUDE_DIRS})

add_executable(single_quadrics_estimation EXCLUDE_FROM_ALL single_quadrics_estimation.cpp)
add_executable(single_quadrics_estimation_with_noise EXCLUDE_FROM_ALL single_quadrics_estimation_with_noise.cpp)
add_executable(single_quadrics_estimation_batch EXCLUDE_FROM_ALL single_quadrics_estimation_batch.cpp)

target_link_libraries(single_quadrics_estimation 
    gtsam 
    ${Python3_LIBRARIES} 
    pybind11::embed
    matplotlibcpp17::matplotlibcpp17;${CONVENIENCE_LIB_NAME})
target_link_libraries(single_quadrics_estimation_with_noise 
    gtsam 
    ${Python3_LIBRARIES} 
    pybind11::embed
    matplotlibcpp17::matplotlibcpp17;${CONVENIENCE_LIB_NAME})
target_link_libraries(single_quadrics_estimation_batch 
    gtsam 
    ${Python3_LIBRARIES} 
    pybind11::embed
    matplotlibcpp17::matplotlibcpp17;${CONVENIENCE_LIB_NAME})

add_custom_target(demo_folder)
add_dependencies(demo_folder 
    single_quadrics_estimation
    single_quadrics_estimation_with_noise
    single_quadrics_estimation_batch
)