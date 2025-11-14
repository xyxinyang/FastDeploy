"""
# Copyright (c) 2025  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""

import contextlib
from typing import Iterable, Any

import paddle
from paddle import nn
from paddleformers.utils.log import logger

from fastdeploy.config import FDConfig, LoadConfig, ModelConfig
from fastdeploy.model_executor.model_loader.base_loader import BaseModelLoader
from fastdeploy.model_executor.models.model_base import ModelRegistry
from fastdeploy.model_executor.utils import process_final_after_loading


def _is_floating_dtype(dtype) -> bool:
    # Helper since paddle does not have a generic is_floating_point for dtypes
    return dtype in (
        paddle.float32,
        paddle.float16,
        getattr(paddle, "bfloat16", None),
        # Some platforms expose float8 types; treat as non-floating here
    )


def initialize_dummy_weights(
    model: nn.Layer,
    low: float = -1e-3,
    high: float = 1e-3,
    seed: int = 1234,
) -> None:
    """
    Initialize all model parameters with fake values to avoid checkpoint IO.

    - Floating-point params: uniform in [low, high].
    - Known scale/bias params (by name suffix): set to 1.0 to avoid degenerate
      divisions during inference.
    - Non-floating params (e.g. int8 quant weights): set to 0.

    The generator is re-seeded per-parameter for deterministic values that do
    not depend on load order. This mirrors vLLM's dummy loader behavior.
    """

    def _iter_parameters(layer: nn.Layer) -> Iterable[tuple[str, Any]]:
        # named_parameters() is available on paddle.nn.Layer
        for name, param in layer.named_parameters():
            yield name, param

    for name, param in _iter_parameters(model):
        # Some parameters may be lazily initialized (e.g., under LazyGuard)
        if hasattr(param, "_is_initialized") and not param._is_initialized():
            param.initialize()

        dtype = param.dtype
        shape = tuple(param.shape)

        # Guard against empty/None shapes
        if not shape:
            continue

        # Heuristic: treat common scale tensors specially
        is_scale_like = any(
            name.endswith(suffix)
            for suffix in (
                "weight_scale",
                "activation_scale",
                "out_scale",
                "weight_scale_inv",
            )
        )

        if _is_floating_dtype(dtype):
            if is_scale_like:
                fill = paddle.ones(shape, dtype=dtype)
                param.set_value(fill)
                continue

            # Re-seed per parameter for deterministic values that only depend
            # on the parameter shape/dtype, not iteration order.
            try:
                numel = 1
                for d in shape:
                    numel *= int(d)
                dtype_sum = sum(ord(c) for c in str(dtype))
                param_seed = (seed * 1315423911 + numel * 2654435761 + dtype_sum) & 0xFFFFFFFF
            except Exception:
                param_seed = seed

            paddle.seed(param_seed)
            rnd = paddle.uniform(shape, min=low, max=high, dtype=dtype)
            param.set_value(rnd)
        else:
            # For integer/quantized params, fill zeros to keep numerically safe
            zeros = paddle.zeros(shape, dtype=dtype)
            param.set_value(zeros)


class DummyModelLoader(BaseModelLoader):
    """Model loader that skips checkpoint IO and fills dummy weights.

    Intended for fast debugging or rapid restarts where model accuracy is not
    important. This significantly reduces start-up time by avoiding weight
    downloads and deserialization.
    """

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        logger.info("Use DummyModelLoader: initialize fake weights instead of loading checkpoints.")

    def download_model(self, model_config: ModelConfig) -> None:
        # Nothing to download in dummy mode
        return

    def load_model(self, fd_config: FDConfig) -> nn.Layer:
        architectures = fd_config.model_config.architectures[0]
        logger.info(f"Starting to create model {architectures} with dummy weights")

        # Mirror DefaultModelLoader's behavior around lazy init for dynamic load
        if fd_config.load_config.dynamic_load_weight:
            # register rl model
            import fastdeploy.rl  # noqa: F401

            if fd_config.speculative_config.model_type != "mtp":
                architectures = architectures.replace("Ernie5ForCausalLM", "Ernie5MoeForCausalLM")
            else:
                architectures = architectures.replace("Ernie5ForCausalLM", "Ernie5MTPForCausalLM")

            architectures = architectures + "RL"
            context: contextlib.AbstractContextManager = paddle.LazyGuard()
        else:
            context = contextlib.nullcontext()

        with context:
            model_cls = ModelRegistry.get_class(architectures)
            model = model_cls(fd_config)

        model.eval()

        # Fill fake weights and allow layers to run their post-load adjustments
        initialize_dummy_weights(model)

        # Some layers (e.g., KVBatchLinear) require a one-time post-process to
        # derive internal tensors from combined weights even when using dummy
        # values. We perform those minimal adjustments here.
        from fastdeploy.model_executor.layers.linear import KVBatchLinear

        for _, sublayer in model.named_sublayers():
            if isinstance(sublayer, KVBatchLinear):
                try:
                    sublayer.process_weights_after_loading()
                except Exception as e:
                    logger.debug(f"KVBatchLinear post-process skipped: {e}")

        # Run generic post-load hooks for layers that expose them
        process_final_after_loading(model, fd_config)
        return model


__all__ = ["DummyModelLoader", "initialize_dummy_weights"]
