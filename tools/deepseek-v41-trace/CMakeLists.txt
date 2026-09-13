set(TARGET llama-deepseek-v41-trace)
add_executable(${TARGET} llama-trace.cpp)
target_link_libraries(${TARGET} PRIVATE llama-common llama vendor::hash ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(PROMPT_TARGET llama-deepseek-v41-prompt-builder)
add_executable(${PROMPT_TARGET} prompt-builder.cpp)
target_link_libraries(${PROMPT_TARGET} PRIVATE llama-common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${PROMPT_TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} ${PROMPT_TARGET} RUNTIME)
endif()
