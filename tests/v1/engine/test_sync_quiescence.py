import asyncio
from types import SimpleNamespace

import pytest

from vllm.outputs import RequestOutput
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.core import EngineCore
from vllm.v1.engine.output_processor import OutputProcessor


class _Request:
    def __init__(self, num_output_tokens: int, *, finished: bool):
        self.num_output_tokens = num_output_tokens
        self._finished = finished

    def is_finished(self) -> bool:
        return self._finished


class _Scheduler:
    def __init__(self):
        self.requests = {
            "active": _Request(17, finished=False),
            "finished_delayed_free": _Request(31, finished=True),
        }
        self.finished_req_ids = {"finished_freed"}

    def get_request_counts(self):
        return (1, 0)

    def has_unfinished_requests(self):
        return True


def test_sync_quiescence_offsets_exclude_finished_delayed_free_requests():
    engine = object.__new__(EngineCore)
    engine.scheduler = _Scheduler()
    engine.batch_queue = []
    engine._scheduler_paused = True
    engine.engine_index = 0
    engine.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(data_parallel_size=1)
    )

    state = engine.get_sync_quiescence_state("unit-test")

    assert state["request_output_token_offsets"] == {"active": 17}
    assert state["terminal_request_ids"] == [
        "finished_delayed_free",
        "finished_freed",
    ]
    assert state["sync_drain_quiesced"] is True


def test_sync_prepare_adds_frontend_lifecycle_receipts():
    class _EngineCoreClient:
        async def prepare_for_sync_async(self, **_kwargs):
            return [
                {
                    "request_output_token_offsets": {"active-deadbeef": 17},
                    "terminal_request_ids": ["finished-in-core-deadbeef"],
                    "sync_drain_quiesced": True,
                }
            ]

        def shutdown(self):
            pass

    async def scenario():
        engine = object.__new__(AsyncLLM)
        engine.vllm_config = SimpleNamespace(
            parallel_config=SimpleNamespace(data_parallel_size=1)
        )
        engine._pause_cond = asyncio.Condition()
        engine._paused = False
        engine.engine_core = _EngineCoreClient()
        engine.output_processor = SimpleNamespace(
            request_states={
                "active-deadbeef": SimpleNamespace(external_req_id="active"),
                "submitted-zero-token-cafebabe": SimpleNamespace(
                    external_req_id="submitted-zero-token"
                ),
            }
        )
        engine._sync_terminal_request_ids = {"terminal-awaiting-caller"}

        states = await engine.prepare_for_sync_generation(reason="unit-test")

        assert states == [
            {
                "request_output_token_offsets": {"active": 17},
                "terminal_request_ids": [],
                "frontend_request_ids": ["active", "submitted-zero-token"],
                "frontend_terminal_request_ids": ["terminal-awaiting-caller"],
                "sync_drain_quiesced": True,
            }
        ]
        assert engine._paused is True

    asyncio.run(scenario())


def test_terminal_receipt_lives_until_generate_caller_consumes_output():
    class _Collector:
        request_id = "terminal-deadbeef"

        def __init__(self, output):
            self.output = output

        def get_nowait(self):
            output, self.output = self.output, None
            return output

        async def get(self):
            raise AssertionError("terminal output should already be available")

        def close(self):
            pass

    async def scenario():
        engine = object.__new__(AsyncLLM)
        engine._sync_terminal_request_ids = set()
        engine.log_requests = False
        output = RequestOutput(
            request_id="terminal",
            prompt=None,
            prompt_token_ids=None,
            prompt_logprobs=None,
            outputs=[],
            finished=True,
        )
        collector = _Collector(output)

        async def add_request(*_args, **_kwargs):
            return collector

        engine.add_request = add_request
        generator = engine.generate(
            prompt=None,
            sampling_params=None,
            request_id="terminal-awaiting-caller",
        )

        assert await anext(generator) is output
        assert engine._sync_terminal_request_ids == {"terminal"}
        with pytest.raises(StopAsyncIteration):
            await anext(generator)
        assert engine._sync_terminal_request_ids == set()

    asyncio.run(scenario())


def test_output_processor_records_terminal_before_publishing_abort():
    class _Queue:
        def __init__(self):
            self.outputs = []

        def put(self, output):
            self.outputs.append(output)

    class _State:
        external_req_id = "external"
        lora_name = None
        detokenizer = object()

        def __init__(self):
            self.queue = _Queue()

        def make_request_output(self, **_kwargs):
            return RequestOutput(
                request_id="external",
                prompt=None,
                prompt_token_ids=None,
                prompt_logprobs=None,
                outputs=[],
                finished=True,
            )

    processor = object.__new__(OutputProcessor)
    state = _State()
    processor.request_states = {"terminal-abort": state}
    processor.external_req_ids = {"external": ["terminal-abort"]}
    processor.parent_requests = {}
    processor.lora_states = SimpleNamespace(
        request_finished=lambda *_args: None
    )
    processor._requests_drained = asyncio.Event()
    processor.sync_terminal_request_ids = set()

    aborted = processor.abort_requests(["terminal-abort"], internal=True)

    assert aborted == ["terminal-abort"]
    assert processor.sync_terminal_request_ids == {"external"}
    assert state.queue.outputs[0].finished is True
