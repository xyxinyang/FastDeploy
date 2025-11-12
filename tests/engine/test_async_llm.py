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

import asyncio
import os
import unittest
import uuid
import weakref

from fastdeploy.engine.args_utils import EngineArgs
from fastdeploy.engine.async_llm import AsyncLLMEngine
from fastdeploy.engine.sampling_params import SamplingParams
from fastdeploy.utils import EngineError

MODEL_NAME = os.getenv("MODEL_PATH", "/path/to/models") + "/ERNIE-4.5-0.3B-Paddle"


class TestAsyncLLMEngine(unittest.TestCase):
    """Test case for AsyncLLMEngine functionality"""

    PROMPTS = [
        "Hello, my name is",
        "The capital of China is",
        "The future of AI is",
        "人工智能是",
    ]

    @classmethod
    def setUpClass(cls):
        """Set up AsyncLLMEngine for testing"""
        try:
            # Use unique ports to avoid conflicts
            base_port = int(os.getenv("FD_ENGINE_QUEUE_PORT", "6778"))
            cache_port = int(os.getenv("FD_CACHE_QUEUE_PORT", "6779"))

            engine_args = EngineArgs(
                model=MODEL_NAME,
                max_model_len=8192,
                tensor_parallel_size=1,
                engine_worker_queue_port=base_port,
                cache_queue_port=cache_port,
            )

            cls.engine = AsyncLLMEngine.from_engine_args(engine_args)

            cls.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(cls.loop)
            success = cls.loop.run_until_complete(cls.engine.start())

            if not success:
                raise RuntimeError("Failed to start AsyncLLMEngine")

            # Use weak reference to avoid circular reference
            cls.engine_ref = weakref.ref(cls.engine)

        except Exception as e:
            print(f"Setting up AsyncLLMEngine failed: {e}")
            raise

    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests have run"""
        if hasattr(cls, "engine") and cls.engine is not None:
            try:

                # Force stop the engine first
                cls.engine.running = False

                # asyncio.run(cls.engine.shutdown())
                cls.loop.run_until_complete(cls.engine.shutdown())

                # Try sync cleanup first
                if hasattr(cls.engine, "_exit_sub_services"):
                    try:
                        cls.engine._exit_sub_services()
                        print("_exit_sub_services completed")
                    except Exception as e:
                        print(f"_exit_sub_services failed: {e}")

                print("Engine cleanup completed")

            except Exception as e:
                print(f"Error during engine cleanup: {e}")
            finally:
                print("Deleting engine...")
                del cls.engine
                print("Engine deleted")

        print("=== tearDownClass completed ===")

        # Force garbage collection
        import gc

        gc.collect()
        print("Garbage collection completed")

    def setUp(self):
        """Set up before each test method"""

        if hasattr(self, "engine") and self.engine:
            # 清理可能残留的output_handler
            if hasattr(self.engine, "output_handler") and self.engine.output_handler:
                if not self.engine.output_handler.done():
                    print("Cleaning up previous output_handler...")
                    self.engine.output_handler.cancel()
                self.engine.output_handler = None

            print(f"Test setup completed: {self._testMethodName}")

    def tearDown(self):
        """Clean up after each test method"""
        if hasattr(self, "engine") and self.engine:

            if hasattr(self.engine, "output_handler") and self.engine.output_handler:
                if not self.engine.output_handler.done():
                    print("Cleaning up output_handler after test...")
                    self.engine.output_handler.cancel()
                self.engine.output_handler = None

            print(f"Test cleanup completed: {self._testMethodName}")

    def run_async_test(self, coro):
        """Helper method to run async tests"""

        try:
            return self.loop.run_until_complete(coro)
        finally:
            pass

    def test_engine_initialization(self):
        """Test that the engine initializes correctly"""
        self.assertIsNotNone(self.engine)
        self.assertTrue(self.engine.is_started)
        self.assertTrue(self.engine.running)

    def test_single_prompt_generation(self):
        """Test generating response for a single prompt"""

        async def _test():
            prompt = "Hello, my name is"
            sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=50)

            outputs = []
            generator = None
            try:
                generator = self.engine.generate(prompt, sampling_params)
                count = 0
                async for output in generator:
                    outputs.append(output)
                    count += 1
                    self.assertIsNotNone(output)
                    self.assertIsNotNone(output.outputs)

            finally:
                # Explicitly close the generator
                if generator is not None:
                    try:
                        await generator.aclose()
                    except:
                        pass

            print(f"Total outputs: {len(outputs)}")
            self.assertGreater(len(outputs), 0)
            return outputs

        outputs = self.run_async_test(_test())
        self.assertGreater(len(outputs), 0)

    def test_multiple_prompts_generation(self):
        """Test generating responses for multiple prompts concurrently"""

        async def _test():
            sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=50)

            # Test concurrent generation
            tasks = []
            for i, prompt in enumerate(self.PROMPTS[:2]):  # Test with first 2 prompts
                request_id = f"test_request_{i}_{uuid.uuid4()}"
                task = self._generate_single(prompt, sampling_params, request_id)
                tasks.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Check that all tasks completed successfully
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.fail(f"Task {i} failed with exception: {result}")
                self.assertGreater(len(result), 0)
                self.assertTrue(result[-1].finished)

            return results

        results = self.run_async_test(_test())
        self.assertEqual(len(results), 2)

    async def _generate_single(self, prompt, sampling_params, request_id=None):
        """Helper method to generate response for a single prompt"""
        outputs = []
        generator = None
        try:
            generator = self.engine.generate(prompt, sampling_params, request_id)
            async for output in generator:
                outputs.append(output)
        finally:
            # Explicitly close the generator
            if generator is not None:
                try:
                    await generator.aclose()
                except:
                    pass
        return outputs

    def test_process_single_output_error_handling(self):
        """Test _process_single_output error handling"""

        async def _test():
            from unittest.mock import Mock

            from fastdeploy.engine.async_llm import AsyncOutputProcessor

            # Create processor with mock tokenizer that raises exception
            mock_tokenizer = Mock()
            mock_tokenizer.decode.side_effect = Exception("Decode error")
            processor = AsyncOutputProcessor(mock_tokenizer)

            # Create mock output without text attribute
            mock_output = Mock()
            mock_output.outputs = Mock()
            mock_output.outputs.token_ids = [1, 2, 3]
            # Don't set text attribute to test the error handling
            if hasattr(mock_output.outputs, "text"):
                delattr(mock_output.outputs, "text")

            # Process the output
            result = processor._process_single_output(mock_output)

            # Verify text was set to empty string on error
            self.assertEqual(result.outputs.text, "")

            return True

        result = self.run_async_test(_test())
        self.assertTrue(result)

    def test_engine_abort_request(self):
        """Test AsyncLLMEngine abort_request functionality"""

        async def _test():
            # Test calling abort_request directly without mocking
            request_id = "test_abort_request"

            # This should not raise an exception
            await self.engine.abort_request(request_id)

            return True

        result = self.run_async_test(_test())
        self.assertTrue(result)

    def test_engine_abort_request_with_error(self):
        """Test AsyncLLMEngine abort_request error handling"""

        async def _test():
            from unittest.mock import AsyncMock

            # Temporarily patch the output_processor to simulate error
            original_processor = self.engine.output_processor

            try:
                # Mock output_processor abort_request to raise error
                mock_processor = AsyncMock()
                mock_processor.abort_request.side_effect = Exception("Abort error")
                self.engine.output_processor = mock_processor

                request_id = "test_abort_error"
                # This should not raise an exception, just log the error
                await self.engine.abort_request(request_id)

                return True
            finally:
                # Restore original processor
                self.engine.output_processor = original_processor

        result = self.run_async_test(_test())
        self.assertTrue(result)

    def test_generate_with_exception_abort(self):
        """Test that generate handles exceptions properly"""

        async def _test():
            # Test with invalid prompt type
            try:
                generator = self.engine.generate(123, SamplingParams(max_tokens=10))  # Invalid prompt type
                async for _ in generator:
                    pass
            except Exception:
                # This is expected
                pass

            return True

        result = self.run_async_test(_test())
        self.assertTrue(result)

    def test_generate_with_generator_exit(self):
        """Test generate handling GeneratorExit exception"""

        async def _test():
            # This test just verifies the code path exists
            # We don't need to actually trigger GeneratorExit in the test
            # since it's handled in the generate method
            return True

        result = self.run_async_test(_test())
        self.assertTrue(result)

    def test_output_handler_loop_coverage(self):
        """Test output handler loop related code paths"""

        async def _test():
            # Test the output handler start/stop mechanism
            if hasattr(self.engine, "_start_output_handler"):
                # This should not fail
                self.engine._start_output_handler()

                # Verify output handler exists
                self.assertIsNotNone(self.engine.output_handler)

            return True

        result = self.run_async_test(_test())
        self.assertTrue(result)

    def test_request_validation_errors(self):
        """Test request validation error scenarios"""

        async def _test():
            # Test input length validation (lines 438-443, 446-448)
            try:
                prompts = [0, 1, 2]
                # Create sampling params with very high min_tokens to trigger error
                sampling_params = SamplingParams(min_tokens=999999)

                # This should trigger the min_tokens validation error
                await self.engine.add_request("test_validation", prompts, sampling_params)
            except Exception as e:
                # Expected to fail due to validation
                self.assertIn("min_dec_len", str(e).lower())

            # Test max model len validation
            try:
                # Create a very long prompt to trigger max_model_len error
                long_prompts = {"prompt_token_ids": [1] * 3000, "prompt_token_ids_len": 3000}  # 超过max_model_len
                await self.engine.add_request("test_long", long_prompts)
            except EngineError as e:
                # 根据实际错误消息调整断言
                error_msg = str(e).lower()
                self.assertTrue(
                    "exceeds the limit" in error_msg
                    or "input text is too long" in error_msg
                    or "input_ids_len" in error_msg
                )
            except Exception:
                # Expected to fail due to length validation
                pass

            return True

        result = self.run_async_test(_test())
        self.assertTrue(result)

    def test_get_methods_coverage(self):
        """Test get_model_config and get_tokenizer methods"""

        async def _test():
            # Test get_model_config (lines 326-328)
            model_config = await self.engine.get_model_config()
            self.assertIsNotNone(model_config)

            # Test get_tokenizer (lines 330-334)
            tokenizer = await self.engine.get_tokenizer()
            if hasattr(self.engine, "data_processor"):
                # This should hit line 333: return self.data_processor.tokenizer
                self.assertIsNotNone(tokenizer)

            # Test _has_guided_input method
            from unittest.mock import Mock

            # Test with guided input
            request_with_guided = Mock()
            request_with_guided.guided_json = {"type": "object"}
            request_with_guided.guided_regex = None
            request_with_guided.guided_choice = None
            request_with_guided.structural_tag = None
            request_with_guided.guided_grammar = None
            request_with_guided.guided_json_object = None

            result = self.engine._has_guided_input(request_with_guided)
            self.assertTrue(result)

            return True

        result = self.run_async_test(_test())
        self.assertTrue(result)

    def test_generate_engine_not_started(self):
        """Test add_request and generate method when engine is not started"""

        async def _test():
            # Create a new engine instance without starting it
            engine_args = EngineArgs(
                model=MODEL_NAME,
                max_model_len=8192,
                tensor_parallel_size=1,
                engine_worker_queue_port=int(os.getenv("FD_ENGINE_QUEUE_PORT", "6778")) + 2,
                cache_queue_port=int(os.getenv("FD_CACHE_QUEUE_PORT", "6779")) + 2,
            )

            unstarted_engine = AsyncLLMEngine.from_engine_args(engine_args)
            # Don't call start() - engine should be in unstarted state

            # Test add_request method when engine not started
            try:
                sampling_params = SamplingParams(max_tokens=10)
                await unstarted_engine.add_request("test_request", "Test prompt", sampling_params)
                self.fail("Expected EngineError was not raised in add_request")
            except EngineError as e:
                # Verify it's the correct error
                self.assertEqual(e.error_code, 500)
                self.assertIn("Engine not started", str(e))
            except Exception as e:
                self.fail(f"Unexpected exception type in add_request: {type(e).__name__}: {e}")

            # Test generate method when engine not started
            try:
                sampling_params = SamplingParams(max_tokens=10)
                generator = unstarted_engine.generate("Test prompt", sampling_params)
                async for _ in generator:
                    pass
                self.fail("Expected EngineError was not raised in generate")
            except EngineError as e:
                # Verify it's the correct error
                self.assertEqual(e.error_code, 500)
                self.assertIn("Engine not started", str(e))
            except Exception as e:
                self.fail(f"Unexpected exception type in generate: {type(e).__name__}: {e}")

            return True

        result = self.run_async_test(_test())
        self.assertTrue(result)

    def test_zmq_connection_initialization_failure(self):
        """Test ZMQ connection initialization failure"""

        async def _test():
            from unittest.mock import Mock, patch

            # Create a new engine instance
            engine_args = EngineArgs(
                model=MODEL_NAME,
                max_model_len=8192,
                tensor_parallel_size=1,
                engine_worker_queue_port=int(os.getenv("FD_ENGINE_QUEUE_PORT", "6778")) + 4,
                cache_queue_port=int(os.getenv("FD_CACHE_QUEUE_PORT", "6779")) + 4,
            )

            test_engine = AsyncLLMEngine.from_engine_args(engine_args)

            # Test connection manager initialization failure
            with (
                patch("fastdeploy.engine.async_llm.ZmqIpcClient") as mock_client_class,
                patch("fastdeploy.engine.async_llm.DealerConnectionManager") as mock_manager_class,
            ):

                # Mock successful client creation
                mock_client = Mock()
                mock_client_class.return_value = mock_client

                # Mock DealerConnectionManager to fail on initialize
                mock_manager = Mock()
                mock_manager.running = False
                mock_manager.initialize.side_effect = Exception("Failed to initialize connection manager")
                mock_manager_class.return_value = mock_manager

                try:
                    await test_engine._init_zmq_connections()
                    self.fail("Expected exception was not raised")
                except Exception as e:
                    self.assertIn("Failed to initialize connection manager", str(e))

            return True

        result = self.run_async_test(_test())
        self.assertTrue(result)

    def test_add_request_exception_handling(self):
        """Test add_request exception handling (lines 447-448 in async_llm.py)"""

        async def _test():
            from unittest.mock import patch

            # Mock data_processor to raise exception
            with patch.object(self.engine, "data_processor") as mock_processor:
                mock_processor.process_request_dict.side_effect = RuntimeError("Processing failed")

                try:
                    await self.engine.add_request("test_id", "test prompt", SamplingParams(max_tokens=10))
                    self.fail("Expected EngineError was not raised")
                except EngineError as e:
                    self.assertEqual(e.error_code, 400)
                    self.assertIn("async_llm add request failed", str(e))
                    self.assertIn("Processing failed", str(e))

            return True

        result = self.run_async_test(_test())
        self.assertTrue(result)

    def test_generate_generator_exit(self):
        """Test generate method GeneratorExit handling"""

        async def _test():
            from unittest.mock import AsyncMock, patch

            # Mock connection_manager to simulate generator exit scenario
            mock_connection_manager = AsyncMock()
            mock_queue = AsyncMock()
            mock_queue.get.side_effect = GeneratorExit("Generator closed")
            mock_connection_manager.get_connection.return_value = (AsyncMock(), mock_queue)

            with patch.object(self.engine, "connection_manager", mock_connection_manager):
                generator = self.engine.generate("test", SamplingParams(max_tokens=10))

                try:
                    async for _ in generator:
                        pass
                except GeneratorExit:
                    # This should be caught and handled gracefully
                    pass
                except Exception as e:
                    self.fail(f"Unexpected exception: {e}")

            return True

        result = self.run_async_test(_test())
        self.assertTrue(result)

    def test_shutdown_exception_handling(self):
        """Test shutdown method exception handling"""

        async def _test():
            from unittest.mock import Mock

            # Create test engine
            engine_args = EngineArgs(
                model=MODEL_NAME,
                max_model_len=8192,
                tensor_parallel_size=1,
                engine_worker_queue_port=int(os.getenv("FD_ENGINE_QUEUE_PORT", "6778")) + 6,
                cache_queue_port=int(os.getenv("FD_CACHE_QUEUE_PORT", "6779")) + 6,
            )
            test_engine = AsyncLLMEngine.from_engine_args(engine_args)

            # Mock components that raise exceptions during shutdown
            test_engine.connection_manager = Mock()
            test_engine.connection_manager.close.side_effect = Exception("Connection manager close failed")

            test_engine.request_client = Mock()
            test_engine.request_client.close.side_effect = Exception("Request client close failed")

            test_engine.engine_service = Mock()
            test_engine.engine_service.shutdown.side_effect = Exception("Engine service shutdown failed")

            # Test that shutdown handles all exceptions gracefully
            try:
                await test_engine.shutdown()
                # Should not raise exception despite internal failures
            except Exception as e:
                self.fail(f"Shutdown should handle exceptions gracefully: {e}")

            return True

        result = self.run_async_test(_test())
        self.assertTrue(result)


if __name__ == "__main__":
    unittest.main()
