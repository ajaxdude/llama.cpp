set(TARGET llama-deepseek-v41-trace)
find_package(Git REQUIRED)
execute_process(
    COMMAND ${GIT_EXECUTABLE} rev-parse HEAD
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE DSV41_BUILD_REVISION
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE DSV41_BUILD_REVISION_RESULT)
string(LENGTH "${DSV41_BUILD_REVISION}" DSV41_BUILD_REVISION_LENGTH)
if(NOT DSV41_BUILD_REVISION_RESULT EQUAL 0 OR
        NOT DSV41_BUILD_REVISION_LENGTH EQUAL 40 OR
        NOT DSV41_BUILD_REVISION MATCHES "^[0-9a-f]+$")
    message(FATAL_ERROR "DeepSeek V4.1 trace exporter requires the exact Git revision")
endif()
add_executable(${TARGET} llama-trace.cpp)
target_link_libraries(${TARGET} PRIVATE llama-common llama vendor::hash ${CMAKE_THREAD_LIBS_INIT} ${CMAKE_DL_LIBS})
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
target_compile_definitions(${TARGET} PRIVATE DSV41_BUILD_REVISION="${DSV41_BUILD_REVISION}")

set(PROMPT_TARGET llama-deepseek-v41-prompt-builder)
add_executable(${PROMPT_TARGET} prompt-builder.cpp)
target_link_libraries(${PROMPT_TARGET} PRIVATE llama-common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${PROMPT_TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} ${PROMPT_TARGET} RUNTIME)
endif()

if(LLAMA_BUILD_TESTS)
    set(MANIFEST_TEST_TARGET test-deepseek41-trace-manifest)
    add_executable(${MANIFEST_TEST_TARGET} llama-trace.cpp)
    target_link_libraries(
        ${MANIFEST_TEST_TARGET}
        PRIVATE llama-common llama vendor::hash ${CMAKE_THREAD_LIBS_INIT} ${CMAKE_DL_LIBS})
    target_include_directories(${MANIFEST_TEST_TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_compile_features(${MANIFEST_TEST_TARGET} PRIVATE cxx_std_17)
    target_compile_definitions(
        ${MANIFEST_TEST_TARGET}
        PRIVATE
            DSV41_BUILD_REVISION="${DSV41_BUILD_REVISION}"
            DSV41_MANIFEST_TEST_HARNESS=1)

    set(INJECTED_TEST_TARGET test-deepseek41-trace-injected)
    add_library(${INJECTED_TEST_TARGET} SHARED test-injected-library.cpp)
    set_target_properties(
        ${INJECTED_TEST_TARGET}
        PROPERTIES
            PREFIX ""
            OUTPUT_NAME "libggml-injected"
            SUFFIX ".module")

    set(HOST_TEST_TARGET test-deepseek41-trace-host)
    add_executable(${HOST_TEST_TARGET} test-host-attestation.cpp)
    target_link_libraries(${HOST_TEST_TARGET} PRIVATE ggml)
    target_compile_features(${HOST_TEST_TARGET} PRIVATE cxx_std_17)
    add_test(NAME ${HOST_TEST_TARGET} COMMAND $<TARGET_FILE:${HOST_TEST_TARGET}>)
    set_property(TEST ${HOST_TEST_TARGET} PROPERTY LABELS main)
endif()
