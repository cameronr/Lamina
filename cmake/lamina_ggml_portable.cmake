# lamina_ggml_portable(): ggml-cpu built for any CPU of the build's architecture rather than this machine's (the
# release binaries, the Docker image, setup's --portable): x86-64 with AVX2 and nothing above it (a 9900K or a Zen 3
# has neither AVX-512 nor AVX-VNNI), arm64 with Armv8.2's dot product and FP16 instructions (Ampere Altra, Grace,
# GB10, Snapdragon X).  The engine (CMakeLists.txt) and the image encoder (tools/vision) both take it.
function(lamina_ggml_portable)
  set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$")
    set(GGML_CPU_ARM_ARCH "armv8.2-a+dotprod+fp16" CACHE STRING "" FORCE)
  else()
    foreach(_f GGML_AVX GGML_AVX2 GGML_FMA GGML_F16C GGML_BMI2)
      set(${_f} ON CACHE BOOL "" FORCE)
    endforeach()
    foreach(_f GGML_AVX512 GGML_AVX512_VBMI GGML_AVX512_VNNI GGML_AVX512_BF16 GGML_AVX_VNNI
               GGML_AMX_TILE GGML_AMX_INT8 GGML_AMX_BF16)
      set(${_f} OFF CACHE BOOL "" FORCE)
    endforeach()
  endif()
endfunction()
