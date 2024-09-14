add_library(th_llama STATIC LlamaOp.cc)
set_property(TARGET th_llama PROPERTY POSITION_INDEPENDENT_CODE  ON)
target_link_libraries(th_llama PRIVATE "${TORCH_LIBRARIES}" Llama th_utils nccl_utils)