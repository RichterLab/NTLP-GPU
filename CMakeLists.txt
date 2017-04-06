cmake_minimum_required (VERSION 2.8)
project (YIP)
enable_language (Fortran)

# Default to Release
if (NOT CMAKE_BUILD_TYPE)
  set (CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: None Debug Release." FORCE)
endif (NOT CMAKE_BUILD_TYPE)

# NAN Verification
option( BUILD_VERIFY_NAN "Build code to test if fields contain NANs" OFF)
if (BUILD_VERIFY_NAN)
  add_definitions(-DBUILD_VERIFY_NAN)
endif (BUILD_VERIFY_NAN)

# Performance Profile
option( BUILD_PERFORMANCE_PROFILE "Build code to test GPU performance" OFF)
if (BUILD_PERFORMANCE_PROFILE)
  add_definitions(-DBUILD_PERFORMANCE_PROFILE)
endif (BUILD_PERFORMANCE_PROFILE)

# CUDA
option( BUILD_CUDA "Build CUDA code" OFF)
if (BUILD_CUDA)
  find_package(CUDA REQUIRED)
  add_definitions(-DBUILD_CUDA)

  set( BUILD_CUDA_BLOCK_THREADS "128" CACHE STRING "Threads per block for CUDA")
  add_definitions(-DCUDA_BLOCK_THREADS=${BUILD_CUDA_BLOCK_THREADS})

  # cuda_add_executable( lescudas "particle_gpu.cu")
  # cuda_add_library( lescuda "particle_gpu.cu")
  set_source_files_properties( particle_gpu.cpp PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ )
  CUDA_COMPILE(PARTICLE_O "particle_gpu.cpp")

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

  # Find Compiler
  execute_process(COMMAND mpif90 -show OUTPUT_VARIABLE MPI_COMPILE_LINE)
  string(REGEX MATCH "^([\\/a-zA-Z]+)\ " FORTRAN_COMPILER_PATH "${MPI_COMPILE_LINE}")
  get_filename_component(FORTRAN_COMPILER ${FORTRAN_COMPILER_PATH} NAME)

  # Setup Compile Flags
  if (FORTRAN_COMPILER MATCHES "ifort")
    set (CMAKE_Fortran_FLAGS "-i4 -r8 -assume byterecl")
    set_source_files_properties( "particle_gpu.F" COMPILE_FLAGS "-free")

    # Handle ulimit issue for intel compiler
    option( BUILD_FORTRAN_RESTRICT_STACK "Restrict stack size for intel compiler" ON)
    if (BUILD_FORTRAN_RESTRICT_STACK)
      set( BUILD_FORTRAN_RESTRICT_STACK_MAX "256" CACHE STRING "Maximum stack size in kilobytes")

      set (CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -heap-arrays ${BUILD_FORTRAN_RESTRICT_STACK_MAX}")
    endif (BUILD_FORTRAN_RESTRICT_STACK)
  elseif (FORTRAN_COMPILER MATCHES "gfortran")
    set (CMAKE_Fortran_FLAGS "-fdefault-real-8 -fdefault-double-8")
    set_source_files_properties( "particle_gpu.F" COMPILE_FLAGS "-ffree-form -ffree-line-length-none")
  else (FORTRAN_COMPILER MATCHES "ifort")
    message ("MPI Compile Line: " ${MPI_COMPILE_LINE})
    message( FATAL_ERROR "Unknown fortran compiler")
  endif (FORTRAN_COMPILER MATCHES "ifort")

  # Build Library
  add_library (fft STATIC "fft.f")

  # Build Executable
  add_executable (lesmpi.a "les.F" "parameters.F" "fields.F" "fftwk.F" "con_data.F" "con_stats.F" "particle.F" "particle_gpu.F" ${PARTICLE_O})
  target_link_libraries (lesmpi.a fft ${CUDA_LIBRARIES})
  set_target_properties( lesmpi.a PROPERTIES LINKER_LANGUAGE Fortran)

  if( NOT BUILD_CUDA )
    add_library(particle_gpu STATIC "particle_gpu.cpp")
    target_link_libraries (lesmpi.a particle_gpu)
  endif( NOT BUILD_CUDA )

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
    cuda_add_executable (les-test "test/main.cpp" "test/utility.cpp" "test/particle.cpp" ${PARTICLE_O})
  else (BUILD_CUDA)
    add_executable (les-test "test/main.cpp" "test/utility.cpp" "test/particle.cpp" "particle_gpu.cpp")
  endif (BUILD_CUDA)

  option( BUILD_TEST_COVERAGE "Enable code coverage calculation" OFF)
  if( BUILD_TEST_COVERAGE )
    set_target_properties(les-test PROPERTIES COMPILE_FLAGS "-fprofile-arcs -ftest-coverage" )
    set_target_properties(les-test PROPERTIES LINK_FLAGS "-fprofile-arcs -ftest-coverage" )
  endif( BUILD_TEST_COVERAGE )

  target_link_libraries (les-test gtest)
endif (BUILD_TESTS)