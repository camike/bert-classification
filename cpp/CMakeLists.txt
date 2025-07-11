cmake_minimum_required(VERSION 3.12)
project(bert_onnx_inference)

# C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 如果使用MinGW，静态链接GCC运行时库
if(MINGW)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static -static-libgcc -static-libstdc++")
    message(STATUS "启用MinGW静态链接所有库")
endif()

# 设置ONNX Runtime路径
set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/lib/onnxruntime")

# 查找ONNX Runtime
if(EXISTS "${ONNXRUNTIME_ROOT}/include" AND EXISTS "${ONNXRUNTIME_ROOT}/lib")
    message(STATUS "找到ONNX Runtime: ${ONNXRUNTIME_ROOT}")
    set(ONNXRUNTIME_INCLUDE_DIRS "${ONNXRUNTIME_ROOT}/include")
    if(WIN32)
        if(MINGW)
            # MinGW需要使用dll进行链接
            set(ONNXRUNTIME_LIBRARIES "${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll")
        else()
            # MSVC使用lib进行链接
            set(ONNXRUNTIME_LIBRARIES "${ONNXRUNTIME_ROOT}/lib/onnxruntime.lib")
        endif()
    else()
        set(ONNXRUNTIME_LIBRARIES "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so")
    endif()
else()
    message(FATAL_ERROR "未找到ONNX Runtime，请设置正确的ONNXRUNTIME_ROOT路径")
endif()

# 头文件路径
include_directories(${ONNXRUNTIME_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# 添加可执行文件 - 只保留实际存在的源文件
add_executable(bert_inference src/bert_inference.cpp)
add_executable(bert_inference_csv src/bert_inference_csv.cpp)
add_executable(bert_inference_int8 src/bert_inference_int8.cpp)
add_executable(bert_inference_int8_csv src/bert_inference_int8_csv.cpp)
add_executable(bert_inference_tiny_int8 src/bert_inference_tiny_int8.cpp)
add_executable(bert_inference_tiny_int8_csv src/bert_inference_tiny_int8_csv.cpp)
add_executable(bert_inference_lmb_csv src/bert_inference_lmb_csv.cpp)
add_executable(bert_inference_lmb_unquant_csv src/bert_inference_lmb_unquant_csv.cpp)

# 链接库
target_link_libraries(bert_inference ${ONNXRUNTIME_LIBRARIES})
target_link_libraries(bert_inference_csv ${ONNXRUNTIME_LIBRARIES})
target_link_libraries(bert_inference_int8 ${ONNXRUNTIME_LIBRARIES})
target_link_libraries(bert_inference_int8_csv ${ONNXRUNTIME_LIBRARIES})
target_link_libraries(bert_inference_tiny_int8 ${ONNXRUNTIME_LIBRARIES})
target_link_libraries(bert_inference_tiny_int8_csv ${ONNXRUNTIME_LIBRARIES})
target_link_libraries(bert_inference_lmb_csv ${ONNXRUNTIME_LIBRARIES})
target_link_libraries(bert_inference_lmb_unquant_csv ${ONNXRUNTIME_LIBRARIES})

# 复制必要的DLL到输出目录（仅Windows）
if(WIN32)
    add_custom_command(TARGET bert_inference POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ONNXRUNTIME_ROOT}/bin/onnxruntime.dll"
        $<TARGET_FILE_DIR:bert_inference>
    )
    add_custom_command(TARGET bert_inference_csv POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ONNXRUNTIME_ROOT}/bin/onnxruntime.dll"
        $<TARGET_FILE_DIR:bert_inference_csv>
    )
    add_custom_command(TARGET bert_inference_int8 POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ONNXRUNTIME_ROOT}/bin/onnxruntime.dll"
        $<TARGET_FILE_DIR:bert_inference_int8>
    )
    add_custom_command(TARGET bert_inference_int8_csv POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ONNXRUNTIME_ROOT}/bin/onnxruntime.dll"
        $<TARGET_FILE_DIR:bert_inference_int8_csv>
    )
    add_custom_command(TARGET bert_inference_tiny_int8 POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ONNXRUNTIME_ROOT}/bin/onnxruntime.dll"
        $<TARGET_FILE_DIR:bert_inference_tiny_int8>
    )
    add_custom_command(TARGET bert_inference_tiny_int8_csv POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ONNXRUNTIME_ROOT}/bin/onnxruntime.dll"
        $<TARGET_FILE_DIR:bert_inference_tiny_int8_csv>
    )
    add_custom_command(TARGET bert_inference_lmb_csv POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ONNXRUNTIME_ROOT}/bin/onnxruntime.dll"
        $<TARGET_FILE_DIR:bert_inference_lmb_csv>
    )
    add_custom_command(TARGET bert_inference_lmb_unquant_csv POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ONNXRUNTIME_ROOT}/bin/onnxruntime.dll"
        $<TARGET_FILE_DIR:bert_inference_lmb_unquant_csv>
    )
    
    # 复制providers_shared DLL（如果存在）
    if(EXISTS "${ONNXRUNTIME_ROOT}/bin/onnxruntime_providers_shared.dll")
        add_custom_command(TARGET bert_inference POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ONNXRUNTIME_ROOT}/bin/onnxruntime_providers_shared.dll"
            $<TARGET_FILE_DIR:bert_inference>
        )
        add_custom_command(TARGET bert_inference_csv POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ONNXRUNTIME_ROOT}/bin/onnxruntime_providers_shared.dll"
            $<TARGET_FILE_DIR:bert_inference_csv>
        )
        add_custom_command(TARGET bert_inference_int8 POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ONNXRUNTIME_ROOT}/bin/onnxruntime_providers_shared.dll"
            $<TARGET_FILE_DIR:bert_inference_int8>
        )
        add_custom_command(TARGET bert_inference_int8_csv POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ONNXRUNTIME_ROOT}/bin/onnxruntime_providers_shared.dll"
            $<TARGET_FILE_DIR:bert_inference_int8_csv>
        )
        add_custom_command(TARGET bert_inference_tiny_int8 POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ONNXRUNTIME_ROOT}/bin/onnxruntime_providers_shared.dll"
            $<TARGET_FILE_DIR:bert_inference_tiny_int8>
        )
        add_custom_command(TARGET bert_inference_tiny_int8_csv POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ONNXRUNTIME_ROOT}/bin/onnxruntime_providers_shared.dll"
            $<TARGET_FILE_DIR:bert_inference_tiny_int8_csv>
        )
        add_custom_command(TARGET bert_inference_lmb_csv POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ONNXRUNTIME_ROOT}/bin/onnxruntime_providers_shared.dll"
            $<TARGET_FILE_DIR:bert_inference_lmb_csv>
        )
        add_custom_command(TARGET bert_inference_lmb_unquant_csv POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ONNXRUNTIME_ROOT}/bin/onnxruntime_providers_shared.dll"
            $<TARGET_FILE_DIR:bert_inference_lmb_unquant_csv>
        )
    endif()
endif()

# 指定输出目录
set_target_properties(bert_inference
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
set_target_properties(bert_inference_csv
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
set_target_properties(bert_inference_int8
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
set_target_properties(bert_inference_int8_csv
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
set_target_properties(bert_inference_tiny_int8
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
set_target_properties(bert_inference_tiny_int8_csv
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
set_target_properties(bert_inference_lmb_csv
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
set_target_properties(bert_inference_lmb_unquant_csv
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# 将模型文件复制到输出目录
add_custom_command(TARGET bert_inference POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory
    $<TARGET_FILE_DIR:bert_inference>/model
)

add_custom_command(TARGET bert_inference POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_CURRENT_SOURCE_DIR}/model/model.onnx"
    $<TARGET_FILE_DIR:bert_inference>/model/
)

add_custom_command(TARGET bert_inference POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_CURRENT_SOURCE_DIR}/model/vocab.txt"
    $<TARGET_FILE_DIR:bert_inference>/model/
)

# 为标准模型CSV版本复制模型文件
add_custom_command(TARGET bert_inference_csv POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory
    $<TARGET_FILE_DIR:bert_inference_csv>/model
)

add_custom_command(TARGET bert_inference_csv POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_CURRENT_SOURCE_DIR}/model/model.onnx"
    $<TARGET_FILE_DIR:bert_inference_csv>/model/
)

add_custom_command(TARGET bert_inference_csv POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_CURRENT_SOURCE_DIR}/model/vocab.txt"
    $<TARGET_FILE_DIR:bert_inference_csv>/model/
)

# 为INT8版本复制模型文件
add_custom_command(TARGET bert_inference_int8 POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory
    $<TARGET_FILE_DIR:bert_inference_int8>/model
)

add_custom_command(TARGET bert_inference_int8 POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_CURRENT_SOURCE_DIR}/model/bert_model_quant.onnx"
    $<TARGET_FILE_DIR:bert_inference_int8>/model/
)

add_custom_command(TARGET bert_inference_int8 POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_CURRENT_SOURCE_DIR}/model/vocab.txt"
    $<TARGET_FILE_DIR:bert_inference_int8>/model/
)

# 为INT8 CSV版本复制模型文件
add_custom_command(TARGET bert_inference_int8_csv POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory
    $<TARGET_FILE_DIR:bert_inference_int8_csv>/model
)

add_custom_command(TARGET bert_inference_int8_csv POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_CURRENT_SOURCE_DIR}/model/bert_model_quant.onnx"
    $<TARGET_FILE_DIR:bert_inference_int8_csv>/model/
)

add_custom_command(TARGET bert_inference_int8_csv POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_CURRENT_SOURCE_DIR}/model/vocab.txt"
    $<TARGET_FILE_DIR:bert_inference_int8_csv>/model/
)

# 为tiny_int8版本复制模型文件
add_custom_command(TARGET bert_inference_tiny_int8 POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory
    $<TARGET_FILE_DIR:bert_inference_tiny_int8>/model
)

add_custom_command(TARGET bert_inference_tiny_int8 POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_CURRENT_SOURCE_DIR}/model/model.quant.onnx"
    $<TARGET_FILE_DIR:bert_inference_tiny_int8>/model/model.quant.onnx
)

add_custom_command(TARGET bert_inference_tiny_int8 POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_CURRENT_SOURCE_DIR}/model/vocab.txt"
    $<TARGET_FILE_DIR:bert_inference_tiny_int8>/model/
)

# 为tiny_int8_csv版本复制模型文件
add_custom_command(TARGET bert_inference_tiny_int8_csv POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory
    $<TARGET_FILE_DIR:bert_inference_tiny_int8_csv>/model
)

add_custom_command(TARGET bert_inference_tiny_int8_csv POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_CURRENT_SOURCE_DIR}/model/model.quant.onnx"
    $<TARGET_FILE_DIR:bert_inference_tiny_int8_csv>/model/model.quant.onnx
)

add_custom_command(TARGET bert_inference_tiny_int8_csv POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_CURRENT_SOURCE_DIR}/model/vocab.txt"
    $<TARGET_FILE_DIR:bert_inference_tiny_int8_csv>/model/
)

# 为lmb_csv版本复制模型文件
add_custom_command(TARGET bert_inference_lmb_csv POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory
    $<TARGET_FILE_DIR:bert_inference_lmb_csv>/model
)

add_custom_command(TARGET bert_inference_lmb_csv POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_CURRENT_SOURCE_DIR}/model/lmb.onnx"
    $<TARGET_FILE_DIR:bert_inference_lmb_csv>/model/lmb.onnx
)

add_custom_command(TARGET bert_inference_lmb_csv POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_CURRENT_SOURCE_DIR}/model/vocab.txt"
    $<TARGET_FILE_DIR:bert_inference_lmb_csv>/model/
)

# 为lmb_unquant_csv版本复制模型文件
add_custom_command(TARGET bert_inference_lmb_unquant_csv POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory
    $<TARGET_FILE_DIR:bert_inference_lmb_unquant_csv>/model
)

add_custom_command(TARGET bert_inference_lmb_unquant_csv POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_CURRENT_SOURCE_DIR}/model/lmb_unquant.onnx"
    $<TARGET_FILE_DIR:bert_inference_lmb_unquant_csv>/model/lmb_unquant.onnx
)

add_custom_command(TARGET bert_inference_lmb_unquant_csv POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_CURRENT_SOURCE_DIR}/model/vocab.txt"
    $<TARGET_FILE_DIR:bert_inference_lmb_unquant_csv>/model/
)