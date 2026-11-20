# One Ninja job pool for every compile and link of the build: half the logical cores, so that the PC stays usable
# while it builds, and no more jobs than the RAM free at configure time holds at LAMINA_JOB_MIB each (nvcc's largest
# files). LAMINA_JOBS sets the number (the logical cores: the fastest build). Generators other than Ninja ignore pools.
# The engine's build (CMakeLists.txt) takes it, before its first target.
#
#   lamina_job_pool()   the pool lamina_jobs, every later target's (CMAKE_JOB_POOL_COMPILE and _LINK in the caller's
#                       scope); LAMINA_JOBS_USED: its size
# nvcc's largest seen: 0.87 GB (ggml-cuda's argsort.cu), the engine's 0.66 GB (its MMQ instances, kv_q8.cu); eight
# compiles at once 2.5 GB in all (two architectures, Windows, working sets every 0.25 s)
set(LAMINA_JOB_MIB 1536)

macro(lamina_job_pool)
  set(LAMINA_JOBS "" CACHE STRING
      "compile and link jobs at once (empty: half the logical cores, fewer when the free RAM holds fewer at ${LAMINA_JOB_MIB} MiB)")
  if(LAMINA_JOBS)
    set(LAMINA_JOBS_USED "${LAMINA_JOBS}")
    set(_lamina_why "LAMINA_JOBS")
  else()
    cmake_host_system_information(RESULT _lamina_free QUERY AVAILABLE_PHYSICAL_MEMORY)
    cmake_host_system_information(RESULT _lamina_cores QUERY NUMBER_OF_LOGICAL_CORES)
    math(EXPR LAMINA_JOBS_USED "${_lamina_cores} / 2")
    math(EXPR _lamina_ram "${_lamina_free} / ${LAMINA_JOB_MIB}")
    if(_lamina_ram LESS LAMINA_JOBS_USED)
      set(LAMINA_JOBS_USED ${_lamina_ram})
    endif()
    if(LAMINA_JOBS_USED LESS 1)
      set(LAMINA_JOBS_USED 1)
    endif()
    set(_lamina_why "half of ${_lamina_cores} logical cores, ${_lamina_free} MiB of RAM free; LAMINA_JOBS sets another number")
  endif()
  set_property(GLOBAL APPEND PROPERTY JOB_POOLS lamina_jobs=${LAMINA_JOBS_USED})
  set(CMAKE_JOB_POOL_COMPILE lamina_jobs)
  set(CMAKE_JOB_POOL_LINK lamina_jobs)
  message(STATUS "Build jobs at once: ${LAMINA_JOBS_USED} (${_lamina_why})")
endmacro()
