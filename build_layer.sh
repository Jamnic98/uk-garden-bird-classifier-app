#!/bin/bash
set -e

rm -rf layer
mkdir -p layer/python/lib/python3.11/site-packages

docker run --rm -it \
  -v "$(pwd)/layer:/layer" \
  quay.io/pypa/manylinux_2_28_x86_64 /bin/bash -c "
    python3.11 -m venv /tmp/venv && \
    source /tmp/venv/bin/activate && \
    pip install --upgrade pip && \
    pip install onnxruntime --target /layer/python/lib/python3.11/site-packages
  "

cd layer
zip -r ../onnxruntime-layer.zip .
cd ..

echo 'Done! Layer written to onnxruntime-layer.zip'
