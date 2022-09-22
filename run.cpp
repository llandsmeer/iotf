#include <cassert>
#include <vector>
#include <iostream>
#include <onnxruntime_cxx_api.h>

const char * model_path = "/tmp/manual_reloop.onnx";

int main() {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
    const auto& api = Ort::GetApi();
    OrtTensorRTProviderOptionsV2* tensorrt_options;

    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);

    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    Ort::Session session(env, model_path, session_options);
    Ort::AllocatorWithDefaultOptions allocator;

     // Input 0 : name=loop_in
     // Input 0 : type=1
     // Input 0 : num_dims=0

    std::vector<const char*> input_node_names = {"loop_in"};
    std::vector<const char*> output_node_names = {"loop_out"};
    std::vector<float> loop_in = {1.0};
    std::vector<int64_t> dims = {1};

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            /* data */
            loop_in.data(),
            1,
            /* shape */
            dims.data(),
            1);
    assert(input_tensor.IsTensor());

    auto output_tensors = session.Run(
            Ort::RunOptions{nullptr},
            input_node_names.data(),
            &input_tensor,
            1 /* one input */,
            output_node_names.data(),
            1 /* one output */);
    assert(output_tensors.size() == 1 && output_tensors.front().IsTensor());
    float* floatarr = output_tensors.front().GetTensorMutableData<float>();

    std::cout << floatarr[0] << std::endl;
}
