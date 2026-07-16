#!/usr/bin/env python3
"""Dump an ML-Agents ONNX model's input/output names + shapes.

Purpose (EXPERIMENT_PLAN.md E13): resolve the two eval-dispatcher unknowns that need the actual
exported graph, in one command, BEFORE trusting any eval numbers:
  - obs input MAPPING/ORDER: which of obs_0/obs_1 is the flat vector vs. the visual grid (conv) --
    ML-Agents sorts sensors by name, so a custom sensor whose name sorts before "VectorSensor_size*"
    takes obs_0; verify here.
  - visual tensor LAYOUT: the grid input's shape reveals NHWC (1,F,8,1) vs NCHW (1,1,F,8) -> set
    ConvDispatcher's TensorShape to match.
  - recurrent_in WIDTH: confirms RecurrentPpoDispatcher's memorySize (128) matches the graph.

Usage:
  ~/mlagents-venv/bin/python scripts/inspect_onnx.py results/<run-id>/ElevatorController.onnx
"""
import sys
import onnx


def shape_of(vi):
    dims = []
    for d in vi.type.tensor_type.shape.dim:
        dims.append(d.dim_param if d.dim_param else d.dim_value)
    return dims


def main(path):
    m = onnx.load(path)
    g = m.graph
    print(f"== {path} ==")
    # Constant/initializer scalars ML-Agents bakes in (memory_size, version_number, etc.)
    inits = {i.name for i in g.initializer}
    print("\n-- INPUTS (name : shape) --  [initializers excluded]")
    for vi in g.input:
        if vi.name in inits:
            continue
        print(f"  {vi.name:28s} {shape_of(vi)}")
    print("\n-- OUTPUTS (name : shape) --")
    for vi in g.output:
        print(f"  {vi.name:28s} {shape_of(vi)}")
    # Surface the baked constant tensors relevant to eval wiring.
    print("\n-- baked constant nodes of interest --")
    for i in g.initializer:
        if i.name in ("memory_size", "version_number", "is_continuous_control",
                      "action_output_shape", "discrete_action_output_shape"):
            vals = list(onnx.numpy_helper.to_array(i).flatten())
            print(f"  {i.name:28s} {vals}")
    print("\nReminders: obs_0/obs_1 order follows sensor-name sort (FloorGrid < VectorSensor_size*);"
          " a visual input shape with a size-8 dim is the (F x 8) grid.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
