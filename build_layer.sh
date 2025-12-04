#!/bin/bash
set -e

rm -rf layer
mkdir -p layer/python/lib/python3.11/site-packages

# List all your dependencies here
REQS="onnxruntime numpy pillow fastapi mangum boto3 jinja2"

docker run --rm -it \
  -v "$(pwd)/layer:/layer" \
  quay.io/pypa/manylinux_2_28_x86_64 /bin/bash -c "
    python3.11 -m venv /tmp/venv && \
    source /tmp/venv/bin/activate && \
    pip install --upgrade pip && \
    pip install $REQS --target /layer/python/lib/python3.11/site-packages
  "

cd layer
zip -r ../requirements.zip .
cd ..

echo 'Done! Layer written to requirements.zip'
