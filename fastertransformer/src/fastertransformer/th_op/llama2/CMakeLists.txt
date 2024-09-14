add_library(th_llama2 STATIC Llama2Op.cc)
set_property(TARGET th_llama2 PROPERTY POSITION_INDEPENDENT_CODE  ON)
target_link_libraries(th_llama2 PRIVATE "${TORCH_LIBRARIES}" LlamaV2 th_utils nccl_utils)