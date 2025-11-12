"""
# Copyright (c) 2025  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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

from __future__ import annotations

import inspect
import os
import signal
import time
import traceback
import uuid
import weakref
from dataclasses import asdict
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

import numpy as np
import zmq

from fastdeploy.engine.args_utils import EngineArgs
from fastdeploy.engine.common_engine import EngineService
from fastdeploy.engine.request import RequestOutput
from fastdeploy.engine.sampling_params import SamplingParams
from fastdeploy.entrypoints.openai.utils import DealerConnectionManager
from fastdeploy.input.preprocess import InputPreprocessor
from fastdeploy.inter_communicator import IPCSignal
from fastdeploy.inter_communicator.zmq_client import ZmqIpcClient
from fastdeploy.metrics.metrics import main_process_metrics
from fastdeploy.utils import EngineError, console_logger, llm_logger


class AsyncOutputProcessor:
    """Async output processor responsible for distributing engine outputs to corresponding request queues"""

    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer

    def _process_single_output(self, output: RequestOutput) -> RequestOutput:
        """Process single output for token decoding"""
        try:
            token_ids = output.outputs.token_ids
            decoded_text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
            output.outputs.text = decoded_text
        except Exception:
            if not hasattr(output.outputs, "text"):
                output.outputs.text = ""
        return output


class EngineServiceProxy:
    """
    Simplified Engine proxy, only responsible for starting common_engine process
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.engine_process = None
        self.engine_pid = os.getpid()
        self._running = False

        llm_logger.info(f"EngineServiceProxy initialized with engine_pid: {self.engine_pid}")

    def start(self):
        """Start engine service proxy"""
        try:
            # Start independent engine process
            self._start_engine_process()

            # Wait for engine to be ready
            if not self._wait_engine_ready():
                raise EngineError("Engine failed to start within timeout", error_code=500)

            self._running = True
            llm_logger.info("EngineServiceProxy started successfully")

        except Exception as e:
            llm_logger.error(f"Failed to start EngineServiceProxy: {e}")
            raise

    def _start_engine_process(self):
        """Start engine process"""
        try:
            import multiprocessing

            self.shutdown_signal = multiprocessing.Value("i", 0)  # 0=running, 1=shutdown

            def run_engine():
                engine = None

                def signal_handler(signum, frame):
                    llm_logger.info(f"Engine process received signal {signum}, initiating shutdown...")
                    if engine:
                        engine.running = False

                # Register signal handlers
                signal.signal(signal.SIGTERM, signal_handler)
                signal.signal(signal.SIGINT, signal_handler)

                try:
                    engine = EngineService(self.cfg, use_async_llm=True)
                    # Start engine with ZMQ service
                    engine.start(async_llm_pid=self.engine_pid)

                    # Keep engine running until shutdown signal is received
                    while self.shutdown_signal.value == 0 and getattr(engine, "running", True):
                        time.sleep(0.5)

                except Exception as e:
                    llm_logger.error(f"Engine process error: {e}, {str(traceback.format_exc())}")
                finally:
                    if engine and hasattr(engine, "_exit_sub_services"):
                        try:
                            engine._exit_sub_services()
                            llm_logger.info("Engine process cleanup completed")
                        except Exception as e:
                            llm_logger.error(f"Error during engine cleanup: {e}")

            self.engine_process = multiprocessing.Process(target=run_engine)
            self.engine_process.start()

            llm_logger.info(f"Started engine process with PID: {self.engine_process.pid}")

        except Exception as e:
            llm_logger.error(f"Failed to start engine process: {e}")
            raise

    def _wait_engine_ready(self) -> bool:
        """Wait for engine and workers to be fully ready"""
        max_wait_time = 180  # seconds
        wait_interval = 1
        elapsed_time = 0

        llm_logger.info("Waiting for engine and workers to be ready...")

        # Use IPC signals to check engine readiness
        # Get the correct suffix
        ipc_suffix = (
            self.cfg.parallel_config.engine_worker_queue_port[0]
            if hasattr(self.cfg, "parallel_config")
            else self.engine_pid
        )

        # Check if loaded_model_signal exists and is ready
        loaded_model_signal = None

        while elapsed_time < max_wait_time:
            # Try to connect to loaded_model_signal
            if loaded_model_signal is None:
                try:
                    loaded_model_signal = IPCSignal(
                        name="loaded_model_signal",
                        array=np.zeros([1], dtype=np.int32),
                        dtype=np.int32,
                        suffix=ipc_suffix,
                        create=False,
                    )
                except:
                    # Signal not ready yet
                    time.sleep(wait_interval)
                    elapsed_time += wait_interval
                    continue

            # Check if workers have loaded models
            if loaded_model_signal.value[0] > 0:
                llm_logger.info("Workers have loaded models successfully")
                # Give ZMQ service more time to fully start
                llm_logger.info("Waiting additional time for ZMQ service to be ready...")
                time.sleep(5)  # Wait for ZMQ service startup + recv_result_handle
                return True

            time.sleep(wait_interval)
            elapsed_time += wait_interval

            if elapsed_time % 10 == 0:  # Log every 10 seconds
                llm_logger.info(f"Waiting for workers to load models... ({elapsed_time}s)")

        llm_logger.error(f"Engine failed to start within {max_wait_time} seconds")
        return False

    def shutdown(self):
        """Shutdown engine service proxy"""
        llm_logger.info("Shutting down EngineServiceProxy...")

        self._running = False

        # Send graceful shutdown signal to engine process
        if hasattr(self, "shutdown_signal"):
            llm_logger.info("Sending shutdown signal to engine process...")
            self.shutdown_signal.value = 1

        # Wait for engine process to shutdown
        if self.engine_process and self.engine_process.is_alive():
            llm_logger.info("Waiting for engine process to shutdown...")
            self.engine_process.terminate()
            self.engine_process.join(timeout=5)
            if self.engine_process.is_alive():
                llm_logger.warning("Force killing engine process...")
                self.engine_process.kill()

        llm_logger.info("EngineServiceProxy shutdown completed")


class AsyncLLMEngine:
    """
    Engine class responsible for managing the Large Language Model (LLM) operations.

    Attributes:
        cfg (Config): Configuration object containing all the parameters.
        cached_generated_tokens (queue.Queue): Queue to store generated tokens.
        scheduler (LocalScheduler or GlobalScheduler): Scheduling tasks.
        input_processor (InputPreprocessor): Preprocessor for input data.
        resource_manager (ResourceManager): Manager for resource allocation.
        token_processor (TokenProcessor): Processor for token generation.
        engine_worker_queue (EngineWorkerQueue): Queue for communication between engine and workers.
        is_started (bool): Flag indicating if the engine has started.
        do_profile (int): Flag indicating if profiling is enabled.
    """

    @classmethod
    def from_engine_args(cls, engine_args: EngineArgs):
        """
        Creates an AsyncLLMEngine from the provided engine arguments.

        Args:
            engine_args (EngineArgs): Engine arguments object.

        Returns:
            AsyncLLMEngine: Instance of the AsyncLLMEngine class.
        """
        # Create the engine configs.
        config = engine_args.create_engine_config()
        # Create the AsyncLLMEngine.
        return cls(cfg=config)

    def __init__(self, cfg):
        """
        Initializes the AsyncLLMEngine with the provided configuration.

        Args:
            cfg (Config): Config object containing all the configuration parameters.
        """
        self.cfg = cfg
        self.running = True
        self.is_started = False

        self.input_processor = InputPreprocessor(
            cfg.model_config,
            cfg.structured_outputs_config.reasoning_parser,
            cfg.limit_mm_per_prompt,
            cfg.mm_processor_kwargs,
            cfg.tool_parser,
        )

        # Use EngineServiceProxy to start engine process
        self.engine_service = EngineServiceProxy(cfg)

        # Create high-performance async connection manager
        self.connection_manager = None
        self.request_client = None

        self.output_processor = AsyncOutputProcessor()

        self._finalizer = weakref.finalize(self, self._exit_sub_services)

        main_process_metrics.set_cache_config_info(obj=self.cfg.cache_config)

    async def start(self):
        """
        Initializes the engine and starts its sub-services.
        """
        assert not self.is_started, "The engine is already started."
        start_time = time.time()

        # Create data processor
        self.data_processor = self.input_processor.create_processor()

        # Update output processor's tokenizer
        if hasattr(self.data_processor, "tokenizer") and self.data_processor.tokenizer:
            self.output_processor.tokenizer = self.data_processor.tokenizer

        # Start engine service proxy (this will start independent engine process)
        self.engine_service.start()

        # Initialize high-performance ZMQ connections
        await self._init_zmq_connections()

        console_logger.info(f"AsyncLLMEngine started with {time.time() - start_time} seconds.")

        self.is_started = True
        return True

    async def _init_zmq_connections(self):
        """Initialize high-performance ZMQ connections"""
        try:
            # Create ZMQ client for sending requests
            self.request_client = ZmqIpcClient(name=self.engine_service.engine_pid, mode=zmq.PUSH)
            self.request_client.connect()

            # Create high-performance async connection manager for receiving responses
            self.connection_manager = DealerConnectionManager(
                pid=self.engine_service.engine_pid, max_connections=int(os.getenv("FD_DEALER_CONNECTIONS", 50))
            )

            if not self.connection_manager.running:
                await self.connection_manager.initialize()

            llm_logger.info("High-performance ZMQ connections initialized successfully")
        except Exception as e:
            llm_logger.error(f"Failed to initialize ZMQ connections: {e}")
            raise

    async def get_model_config(self):
        """Get model configuration"""
        return self.cfg.model_config

    async def get_tokenizer(self):
        """Get tokenizer"""
        if hasattr(self, "data_processor"):
            return self.data_processor.tokenizer
        return None

    def _has_guided_input(self, request):
        """
        Check if the request has any guided input.
        """
        return any(
            x is not None
            for x in (
                request.guided_json,
                request.guided_regex,
                request.guided_choice,
                request.structural_tag,
                request.guided_grammar,
                request.guided_json_object,
            )
        )

    async def add_request(
        self,
        request_id: str,
        prompt: Union[str, List[str], Dict[str, Any]],
        sampling_params: Optional[SamplingParams] = None,
        arrival_time: Optional[float] = None,
        **kwargs,
    ):
        """
        Async add request

        Args:
            request_id: Request ID
            prompt: Input prompt
            sampling_params: Sampling parameters
            arrival_time: Arrival time
            **kwargs: Other parameters

        """
        if not self.is_started or self.engine_service is None:
            raise EngineError("Engine not started. Call start() first.", error_code=500)

        if request_id is None:
            request_id = str(uuid.uuid4())

        if arrival_time is None:
            arrival_time = time.time()

        if isinstance(prompt, str):
            prompt = {
                "prompt": prompt,
                "request_id": request_id,
            }
        elif isinstance(prompt, list) and isinstance(prompt[0], int):
            prompt = {
                "prompt_token_ids": prompt,
                "request_id": request_id,
            }
        elif isinstance(prompt, dict):
            prompt["request_id"] = request_id
        else:
            raise TypeError(f"Invalid type for 'prompt': {type(prompt)}, expected one of ['str', 'list', 'dict'].")

        if sampling_params is not None:
            prompt.update(asdict(sampling_params))

        try:
            # Check if already preprocessed by api_server
            is_preprocessed = prompt.get("_preprocessed", False)

            if inspect.iscoroutinefunction(self.data_processor.process_request_dict):
                request = await self.data_processor.process_request_dict(prompt, self.cfg.model_config.max_model_len)
            else:
                request = self.data_processor.process_request_dict(prompt, self.cfg.model_config.max_model_len)

            request["prompt_token_ids_len"] = len(request["prompt_token_ids"])

            if not is_preprocessed:
                request["preprocess_start_time"] = arrival_time
                input_ids_len = request["prompt_token_ids_len"]

                request["max_tokens"] = min(
                    self.cfg.model_config.max_model_len - input_ids_len, request.get("max_tokens")
                )

                min_tokens = request.get("min_tokens", 1)
                if input_ids_len + min_tokens >= self.cfg.model_config.max_model_len:
                    error_msg = (
                        f"Input text is too long, length of prompt token({input_ids_len}) "
                        f"+ min_dec_len ({min_tokens}) >= max_model_len "
                    )
                    llm_logger.error(error_msg)
                    raise EngineError(error_msg, error_code=400)

                request["preprocess_end_time"] = time.time()
                preprocess_cost_time = request["preprocess_end_time"] - request["preprocess_start_time"]
                llm_logger.info(
                    f"Cache request with request_id ({request.get('request_id')}), "
                    f"preprocess time cost {preprocess_cost_time}"
                )

            if not self.cfg.model_config.enable_mm:
                self.request_client.send_json(request)
            else:
                self.request_client.send_pyobj(request)

        except EngineError:
            raise
        except Exception as e:
            raise EngineError(f"async_llm add request failed: {e}", error_code=400)

    async def generate(
        self,
        prompt: Union[str, List[str], Dict[str, Any]],
        sampling_params: Optional[SamplingParams] = None,
        request_id: Optional[str] = None,
        **kwargs,
    ) -> AsyncGenerator[RequestOutput, None]:
        """
        Async generation interface

        Args:
            prompt: Input prompt
            sampling_params: Sampling parameters
            request_id: Request ID
            **kwargs: Other parameters

        Yields:
            RequestOutput: Generated output
        """
        if not self.is_started:
            raise EngineError("Engine not started. Call start() first.", error_code=500)

        if request_id is None:
            request_id = str(uuid.uuid4())

        try:
            # Async add request
            await self.add_request(request_id, prompt, sampling_params, **kwargs)

            dealer, response_queue = await self.connection_manager.get_connection(request_id, num_choices=1)

            dealer.write([b"", request_id.encode("utf-8")])

            finished = False
            while not finished:
                # Get response from DealerConnectionManager's queue
                response_list = await response_queue.get()

                # Process response list
                for response_item in response_list:
                    if isinstance(response_item, dict) and "request_id" in response_item:
                        # Convert dict response to RequestOutput object
                        if isinstance(response_item, dict):
                            request_output = RequestOutput.from_dict(response_item)
                        else:
                            request_output = response_item

                        # Use output_processor for token decoding etc.
                        if hasattr(self, "output_processor") and self.output_processor.tokenizer:
                            processed_output = self.output_processor._process_single_output(request_output)
                        else:
                            processed_output = request_output

                        finished = processed_output.finished
                        yield processed_output

                        if finished:
                            break

            await self.connection_manager.cleanup_request(request_id)

        except GeneratorExit:
            llm_logger.info(f"Request {request_id} generator exit (outer)")
            return
        except Exception as e:
            await self.abort_request(request_id)
            llm_logger.error(f"Request {request_id} failed: {e}")
            raise EngineError(str(e), error_code=500) from e

    async def abort_request(self, request_id: str) -> None:
        """
        Abort the specified request

        Args:
            request_id: Request ID to abort
        """
        try:
            # Clean up request through DealerConnectionManager
            if hasattr(self, "connection_manager") and self.connection_manager:
                await self.connection_manager.cleanup_request(request_id)
            llm_logger.info(f"Aborted request {request_id}")
        except Exception as e:
            llm_logger.error(f"Failed to abort request {request_id}: {e}")

    async def shutdown(self):
        """
        Gracefully shutdown AsyncLLM engine
        """
        llm_logger.info("Starting AsyncLLM shutdown...")

        self.running = False

        # Close high-performance connection manager
        if hasattr(self, "connection_manager") and self.connection_manager is not None:
            llm_logger.info("Stopping connection manager...")
            try:
                await self.connection_manager.close()
            except Exception as e:
                llm_logger.error(f"Error while stopping connection manager: {e}")

        # Close ZMQ client
        if hasattr(self, "request_client") and self.request_client is not None:
            llm_logger.info("Closing request client...")
            try:
                self.request_client.close()
            except Exception as e:
                llm_logger.warning(f"Error closing request client: {e}")

        # Shutdown engine service proxy
        if hasattr(self, "engine_service") and self.engine_service is not None:
            llm_logger.info("Stopping engine service proxy...")
            try:
                self.engine_service.shutdown()
            except Exception as e:
                llm_logger.error(f"Error while stopping engine service proxy: {e}")

        self.is_started = False
        llm_logger.info("AsyncLLM shutdown completed")

    def _exit_sub_services(self):
        """
        Clean up any remaining resources
        """
        pass
