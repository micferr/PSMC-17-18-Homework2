find_package(OpenCL REQUIRED)
add_executable(test_tgt src/main.c src/pgm.h src/pgm.c)
target_link_libraries(test_tgt OpenCL::OpenCL)