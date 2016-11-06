cmake_minimum_required (VERSION 2.8)
project (YIP)
enable_language (Fortran)

# Default to Release
if (NOT CMAKE_BUILD_TYPE)
  set (CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: None Debug Release." FORCE)
endif (NOT CMAKE_BUILD_TYPE)

# CUDA
option( BUILD_CUDA "Build CUDA code" OFF)
if (BUILD_CUDA)
  find_package(CUDA REQUIRED)
  add_definitions(-DBUILD_CUDA)

  # cuda_add_executable( lescudas "particle_gpu.cu")
  # cuda_add_library( lescuda "particle_gpu.cu")
  CUDA_COMPILE(PARTICLE_O "particle_gpu.cu")

  option( BUILD_CUDA_VERIFY "Build CUDA verification code" OFF)
  if(BUILD_CUDA_VERIFY)
    add_definitions(-DBUILD_CUDA_VERIFY)
  endif(BUILD_CUDA_VERIFY)

  # target_link_libraries (lesmpi.a lescuda)
endif(BUILD_CUDA)

option( BUILD_FORTRAN "Build Fortran code" ON)
if (BUILD_FORTRAN)
  # Require MPI
  find_package(MPI REQUIRED)
  set( CMAKE_Fortran_COMPILER ${MPI_Fortran_COMPILER})

  # Setup Compile Flags
  set (CMAKE_Fortran_FLAGS_RELEASE "${CMAKE_Fortran_FLAGS_RELEASE} -i4 -r8")
  set (CMAKE_Fortran_FLAGS_DEBUG   "${CMAKE_Fortran_FLAGS_DEBUG} -i4 -r8")

  # Handle ulimit issue for intel compiler
  option( BUILD_FORTRAN_RESTRICT_STACK "Restrict stack size for intel compiler" ON)
  if (BUILD_FORTRAN_RESTRICT_STACK)
    set( BUILD_FORTRAN_RESTRICT_STACK_MAX "256" CACHE STRING "Maximum stack size in kilobytes")

    set (CMAKE_Fortran_FLAGS_RELEASE "${CMAKE_Fortran_FLAGS_RELEASE} -heap-arrays ${BUILD_FORTRAN_RESTRICT_STACK_MAX}")
    set (CMAKE_Fortran_FLAGS_DEBUG   "${CMAKE_Fortran_FLAGS_DEBUG} -heap-arrays ${BUILD_FORTRAN_RESTRICT_STACK_MAX}")
  endif (BUILD_FORTRAN_RESTRICT_STACK)

  # Build Library
  add_library (fft STATIC "fft.f")

  # Build Executable
  add_executable (lesmpi.a "les.F" "parameters.F" "fields.F" "fftwk.F" "con_data.F" "con_stats.F" "particle.F" "particle_gpu.F" ${PARTICLE_O})
  target_link_libraries (lesmpi.a fft ${CUDA_LIBRARIES})

  install ( TARGETS lesmpi.a RUNTIME DESTINATION "bin")
endif (BUILD_FORTRAN)

option( BUILD_TESTS "Build LES tests" OFF)
if (BUILD_TESTS)
  # Download and unpack googletest at configure time
  configure_file(CMakeLists.txt.in googletest-download/CMakeLists.txt)
  execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" . WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/googletest )
  execute_process(COMMAND ${CMAKE_COMMAND} --build . WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/googletest )

  # Prevent GoogleTest from overriding our compiler/linker options
  # when building with Visual Studio
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

  # Add googletest directly to our build. This adds
  # the following targets: gtest, gtest_main, gmock
  # and gmock_main
  add_subdirectory(test/google ${CMAKE_BINARY_DIR}/googletest-build)

  # The gtest/gmock targets carry header search path
  # dependencies automatically when using CMake 2.8.11 or
  # later. Otherwise we have to add them here ourselves.
  if (CMAKE_VERSION VERSION_LESS 2.8.11)
    include_directories("${gtest_SOURCE_DIR}/include")
  endif()

  include_directories("${CMAKE_SOURCE_DIR}" "test/")

  if (BUILD_CUDA)
    cuda_include_directories("${CMAKE_SOURCE_DIR}" "${gtest_SOURCE_DIR}/include" "test/")
    cuda_add_executable (les-test "test/main.cpp" "test/utility.cpp" "test/reference.cpp" "test/cuda.cpp" "particle_gpu.cu")
  else (BUILD_CUDA)
    add_executable (les-test "test/main.cpp" "test/utility.cpp" "test/reference.cpp")
  endif (BUILD_CUDA)
  target_link_libraries (les-test gtest)
endif (BUILD_TESTS)