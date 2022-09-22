set -ex
g++ \
    -I/home/llandsmeer/.local/include/onnx \
    run.cpp \
    -o /tmp/run.x \
    -L/home/llandsmeer/.local/lib \
    -lonnxruntime
/tmp/run.x
