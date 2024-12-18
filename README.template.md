# Inferior olive benchmark

Codebase for the paper

> Landsmeer, Lennart PL, et al. "Tricking AI chips into simulating the human brain: A detailed performance analysis." Neurocomputing (2024): 127953.

## Requirements

 - `numpy`
 - `matplotlib`
 - `tensorflow` - make sure you are using the right version if you want to run on CUDA
 - `onnxruntime`
 - `onnxruntime-gpu` (optional) - to run on CUDA and TensorRT
 - Graphcore tensorflow SDK (optional) - to run on Graphcore
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

DOCS

## Authors

 - Max Engelen
 - Lennart Landsmeer
