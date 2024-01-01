import argparse as ap
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import onnxsim
import torch
from torch import nn

from source.utils.general import get_object_from_dict, read_config

ATOL = 1e-4


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
        type=str,
        help="Path to .YAML configuration file",
    )
    parser.add_argument(
        "--torch_weights",
        required=True,
        type=str,
        help="Path to PyTorch .pth or .pt file",
    )
    parser.add_argument(
        "--onnx_path",
        required=True,
        type=str,
        help="Path for exported onnx file",
    )
    parser.add_argument(
        "--image_size",
        required=True,
        type=str,
        help="Image size to work with",
    )
    args = parser.parse_args()

    config = read_config(Path(args.config))
    model: nn.Module = get_object_from_dict(config.model)
    model.load_state_dict(torch.load(args.torch_weights))
    model.eval()

    image_size = tuple(map(int, args.image_size.split(",")))
    dummy_input = torch.zeros(1, 3, *image_size, dtype=torch.float)
    torch.onnx.export(
        model,
        dummy_input,
        args.onnx_path,
        verbose=True,
        input_names=["input_image"],
        output_names=["output_dict"],
        do_constant_folding=True,
    )

    # Simplify ONNX model.
    onnx_model = onnx.load(args.onnx_path)
    onnx_model, check = onnxsim.simplify(onnx_model, check_n=10)
    save_path = Path(args.onnx_path).parent / (Path(args.onnx_path).name.split(".")[0] + "_sim" + ".onnx")
    onnx.save(onnx_model, save_path)

    # Remove initializers from ONNX model.
    onnx_model = onnx.load(save_path)
    inputs = onnx_model.graph.input
    name_to_input = {input.name: input for input in inputs}
    for initializer in onnx_model.graph.initializer:
        if initializer.name in name_to_input:
            inputs.remove(name_to_input[initializer.name])
    onnx.save(onnx_model, save_path)

    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 4
    sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    ort_session = ort.InferenceSession(save_path, sess_options)

    for _ in range(10):
        random_tensor = np.random.randn(1, 3, *image_size).astype(np.float32)
        onnx_output = ort_session.run(None, {"input_image": random_tensor})[0][0]
        with torch.no_grad():
            torch_output = model(torch.from_numpy(random_tensor).float()).numpy()
        assert np.allclose(onnx_output, torch_output, atol=ATOL)
