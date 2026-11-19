cmake_minimum_required(VERSION 3.1)

find_package(deal.II 9.3.0 REQUIRED
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR})
deal_ii_initialize_cached_variables()
project(npsat CXX)

# This executable uses Trilinos wrappers directly; PETSc alone is insufficient.
if(NOT DEAL_II_WITH_MPI OR NOT DEAL_II_WITH_TRILINOS OR NOT DEAL_II_WITH_P4EST)
  message(FATAL_ERROR "NPSAT requires deal.II with MPI, Trilinos and p4est.")
endif()

# Aqua can supply BOOST_ROOT to match the Boost used to build deal.II.
find_package(Boost REQUIRED COMPONENTS program_options)
add_executable(npsat npsat.cc)
target_include_directories(npsat PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/myheaders ${Boost_INCLUDE_DIRS})
deal_ii_setup_target(npsat)
target_link_libraries(npsat ${Boost_LIBRARIES})
# deal.II 9.3 headers require C++14. Keep GCC 5 as the deployment target;
# target properties survive deal.II's initialization of cached compiler flags.
set_target_properties(npsat PROPERTIES
  CXX_STANDARD 14 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS NO)
