#!/bin/bash

LOG_DIR=unet_trained_bgs_example_data
TF_ROOT=../../tensorflow
TF_BIN=$TF_ROOT/bazel-bin
TF_QUANT_ROOT=../../tensorflow-quantization/tools

# Train the model
./read_radio.py

# At first, freeze the graph from the training log directory
$TF_QUANT_ROOT/freeze_graph.py --log_dir=$LOG_DIR

# Quantize the graph
$TF_BIN/tensorflow/tools/quantization/quantize_graph \
  --input ./unet_trained_bgs_example_data/frozen_model.pb \
  --output_node_names='cross_entropy' \
  --output quantized_model.pb \
  --mode=eightbit

# Evaluate
./eval_frozen_graph.py --frozen_model=$LOG_DIR/frozen_model.pb
./eval_frozen_graph.py --frozen_model=quantized_model.pb

