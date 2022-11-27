# Inferior olive benchmark

## Requirements

 - `numpy`
 - `matplotlib`
 - `tensorflow` - make sure you are using the right version if you want to run on CUDA
 - `onnxruntime`
 - `onnxruntime-gpu` (optional) - to run on CUDA and TensorRT
 - Graphcore tensorflow SDK (optional) - to run on Graphcore
 - GroqWare SDK (optional) - to run on Groq
 - `tf2onnx`

## Usage

Running benchmarks (creating a benchmark.txt file).
Which Runners to benchmark is chosen at runtime based on available hardware.

```
$ git clone https://github.com/llandsmeer/iotf
$ cd iotf
$ python3 -m ioperf
```

Runnings tests (with optional graphcore tests specialization)

```
$ pytest ioperf --disable-warnings [-k graphcore]
[..]
2 passed, 11 deselected, 3 warnings in 6.03s
```

After making changes, run `bash build-docs.sh` and commit

## Module documentation

 - [docs](https://llandsmeer.github.io/iotf/index.html)
     - [model](https://llandsmeer.github.io/iotf/model/index.html)
     - [bench](https://llandsmeer.github.io/iotf/bench/index.html)
         - [benchmark](https://llandsmeer.github.io/iotf/bench/benchmark.html)
         - [model_configuration](https://llandsmeer.github.io/iotf/bench/model_configuration.html)
         - [tests](https://llandsmeer.github.io/iotf/bench/tests/index.html)
             - [test_model_configuration](https://llandsmeer.github.io/iotf/bench/tests/test_model_configuration.html)
     - [runners](https://llandsmeer.github.io/iotf/runners/index.html)
         - [groqchip_runner_opt2_nocopy](https://llandsmeer.github.io/iotf/runners/groqchip_runner_opt2_nocopy.html)
         - [onnx_tensorrt_runner](https://llandsmeer.github.io/iotf/runners/onnx_tensorrt_runner.html)
         - [groqchip_runner_opt1](https://llandsmeer.github.io/iotf/runners/groqchip_runner_opt1.html)
         - [base_runner](https://llandsmeer.github.io/iotf/runners/base_runner.html)
         - [groqchip_runner](https://llandsmeer.github.io/iotf/runners/groqchip_runner.html)
         - [onnx_cpu_runner](https://llandsmeer.github.io/iotf/runners/onnx_cpu_runner.html)
         - [onnx_cuda_runner](https://llandsmeer.github.io/iotf/runners/onnx_cuda_runner.html)
         - [tests](https://llandsmeer.github.io/iotf/runners/tests/index.html)
             - [test_graphcore_runner_with_shim](https://llandsmeer.github.io/iotf/runners/tests/test_graphcore_runner_with_shim.html)
             - [test_groqchip_runner](https://llandsmeer.github.io/iotf/runners/tests/test_groqchip_runner.html)
             - [test_base_runner](https://llandsmeer.github.io/iotf/runners/tests/test_base_runner.html)
             - [test_onnx_cpu_runner](https://llandsmeer.github.io/iotf/runners/tests/test_onnx_cpu_runner.html)
         - [onnx_base_runner](https://llandsmeer.github.io/iotf/runners/onnx_base_runner.html)
         - [graphcore_runner](https://llandsmeer.github.io/iotf/runners/graphcore_runner.html)

## Authors

 - Max Engelen
 - Lennart Landsmeer
