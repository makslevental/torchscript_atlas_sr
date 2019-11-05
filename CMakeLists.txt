cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(main)
set(Torch_DIR libtorch/share/cmake/Torch)
set(OpenCV_DIR opencv/build)
find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)


add_executable(main main.cpp)
target_link_libraries(main "${TORCH_LIBRARIES}")
target_link_libraries(main ${OpenCV_LIBS})

set_property(TARGET main PROPERTY CXX_STANDARD 11)

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET main.cpp
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${TORCH_DLLS}
            $<TARGET_FILE_DIR:main>)
endif (MSVC)

include_directories("cxxopts-2.2.0/include")
