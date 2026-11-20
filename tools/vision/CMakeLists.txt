# tools/vision - lamina-vision, the image encoder of Lamina's multimodal path as a program: the engine's own encoder
# (src/vision, include/lamina/vision) on ggml, from the llama.cpp checkout the engine's ggml comes from
# (cmake/lamina_llama.cmake), configured as that checkout's own build configures it.
#
# The engine's build (../../CMakeLists.txt) builds it as an external project, with CUDA for the engine's GPUs, into its
# build folder beside lamina-server, which starts the lamina-vision in its own folder and no other. A CUDA build runs
# either way, as the config's vision section says: without --gpu it hides the GPUs and runs on the CPU.
# LAMINA_VISION_CUDA=OFF (the default here) builds one for the CPU alone, which refuses --gpu (src/docs/SETUP.md, A
# CPU-only image encoder):
#
#   cmake -S tools/vision -B build-vision -G Ninja -DCMAKE_BUILD_TYPE=Release
#   cmake --build build-vision --target lamina-vision          (build-vision/bin/lamina-vision)
#
# Without LLAMA_DIR it takes the engine's third_party/llama.cpp, downloaded at the pinned commit when it is not there.
# Portable (AVX2 baseline; arm64: Armv8.2 + dotprod + fp16) is the default, so an encoder built without extra flags
# runs on any AVX2 PC: the 0.1.32 release's encoder was built native on an AVX-512 machine and stopped with an illegal
# instruction elsewhere (#411 #412 #419); a native build on an AVX-512 CPU also runs without flash attention on the
# CPU, about half as fast. The option is kept in CMakeCache.txt, so a build folder configured before this default
# still needs the flag (or a fresh folder).
cmake_minimum_required(VERSION 3.21)
project(lamina_vision C CXX)

set(LLAMA_DIR "" CACHE PATH
    "llama.cpp checkout (empty: the engine's third_party/llama.cpp at the pinned commit, downloaded when missing)")
if(NOT LLAMA_DIR)
  include(${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/lamina_llama.cmake)
  lamina_llama_cpp(LLAMA_DIR)
endif()
set(LAMINA_VISION_OUTPUT_DIR "${CMAKE_BINARY_DIR}/bin" CACHE PATH "the folder lamina-vision is built into")
option(LAMINA_VISION_CUDA "run the vision encoder on the GPU (builds ggml-cuda)" OFF)
option(LAMINA_PORTABLE "build for other PCs: AVX2 (arm64: Armv8.2 + dotprod + fp16) baseline, static C++ runtime" ON)
if(LAMINA_PORTABLE AND MSVC)
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded")
elseif(LAMINA_PORTABLE AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
  add_link_options(-static-libstdc++ -static-libgcc -Wl,--exclude-libs,ALL)   # as the engine's CMakeLists.txt says
endif()
# cmake --install: flat, as a release lays its programs out (the engine's build installs it with them)
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "the folder cmake --install fills (--prefix names another)" FORCE)
endif()

# ggml through the checkout's own top level, for the settings it gives ggml there (its CPU products' kernels, the CUDA
# graphs); of its targets only ggml's libraries are built
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_MTMD OFF CACHE BOOL "" FORCE)
set(LLAMA_OPENSSL OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
set(GGML_CCACHE OFF CACHE BOOL "" FORCE)   # no launcher of ggml's choosing: CMAKE_<LANG>_COMPILER_LAUNCHER names one
if(LAMINA_PORTABLE)
  include(${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/lamina_ggml_portable.cmake)
  lamina_ggml_portable()   # AVX2 and nothing above it (arm64: Armv8.2 + dotprod + fp16), even when a -D asks for more
  set(GGML_OPENMP OFF CACHE BOOL "" FORCE)        # no vcomp140.dll / libgomp needed on the user's PC
  message(STATUS "lamina-vision: portable build (runs on other PCs)")
else()
  # this PC's instruction set; a build folder that was portable before gets a fresh native build's settings back
  set(GGML_NATIVE ON CACHE BOOL "" FORCE)
  set(GGML_OPENMP ON CACHE BOOL "" FORCE)
  foreach(_f GGML_AVX GGML_AVX2 GGML_FMA GGML_F16C GGML_BMI2)
    set(${_f} OFF CACHE BOOL "" FORCE)
  endforeach()
  message(STATUS "lamina-vision: native build for this CPU (LAMINA_PORTABLE=OFF) - not for other PCs")
endif()
set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
set(GGML_CUDA ${LAMINA_VISION_CUDA} CACHE BOOL "" FORCE)
# every compile and link in one job pool: half the logical cores (LAMINA_JOBS sets another number)
include(${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/lamina_jobs.cmake)
lamina_job_pool()

# llama.cpp's install rules are its libraries' and headers', none of the encoder's
set(CMAKE_SKIP_INSTALL_RULES ON)
add_subdirectory(${LLAMA_DIR} llama EXCLUDE_FROM_ALL)
set(CMAKE_SKIP_INSTALL_RULES OFF)

set(_lamina_src ${CMAKE_CURRENT_SOURCE_DIR}/../../src/vision)
add_executable(lamina-vision lamina_vision.cpp ${_lamina_src}/image.cpp ${_lamina_src}/ggml_encoder.cpp)
if(MSVC)   # UTF-8 as its code page: image and model paths outside the ANSI code page (../../cmake/utf8.manifest)
  target_sources(lamina-vision PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/utf8.manifest)
endif()
# stb_image from the checkout's vendor folder ("stb/stb_image.h")
target_include_directories(lamina-vision PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../include ${LLAMA_DIR}/vendor)
target_link_libraries(lamina-vision PRIVATE ggml)
target_compile_features(lamina-vision PRIVATE cxx_std_17)
# the engine's warning level: these sources are its own
if(MSVC)
  target_compile_options(lamina-vision PRIVATE /W4 /utf-8)
  target_compile_definitions(lamina-vision PRIVATE _CRT_SECURE_NO_WARNINGS)
else()
  target_compile_options(lamina-vision PRIVATE -Wall -Wextra)
endif()
# a build without CUDA refuses --gpu (lamina_vision.cpp)
target_compile_definitions(lamina-vision PRIVATE LAMINA_VISION_CUDA=$<BOOL:${LAMINA_VISION_CUDA}>)
# the version --version prints: the engine's build passes its own; this folder built alone reads the engine's
# project VERSION (and configures again when that file changes)
if(NOT LAMINA_VERSION)
  set(_lamina_top ${CMAKE_CURRENT_SOURCE_DIR}/../../CMakeLists.txt)
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${_lamina_top})
  file(STRINGS ${_lamina_top} _lamina_project REGEX "^project\\(lamina VERSION ")
  string(REGEX REPLACE "^project\\(lamina VERSION ([0-9.]+).*" "\\1" LAMINA_VERSION "${_lamina_project}")
endif()
set_property(SOURCE lamina_vision.cpp APPEND PROPERTY COMPILE_DEFINITIONS LAMINA_VERSION="${LAMINA_VERSION}")
install(TARGETS lamina-vision RUNTIME DESTINATION . COMPONENT lamina)
set_target_properties(lamina-vision PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${LAMINA_VISION_OUTPUT_DIR})
if(UNIX AND NOT APPLE)
  # installed, it loads the libraries in its own folder (a release's folder is relocatable), in the build folder the
  # toolkit's; RUNPATH, so LD_LIBRARY_PATH can name others
  set_target_properties(lamina-vision PROPERTIES INSTALL_RPATH "$ORIGIN")
  target_link_options(lamina-vision PRIVATE -Wl,--enable-new-dtags)
endif()
