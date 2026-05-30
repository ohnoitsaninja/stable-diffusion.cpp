set(TARGET sd-bonsai-gemlite-int1-gen-smoke)
add_executable(${TARGET} main.cpp)
target_link_libraries(${TARGET} PRIVATE stable-diffusion)
