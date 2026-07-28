from types import SimpleNamespace

from vllm.v1.engine.core import EngineCore


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
    assert state["sync_drain_quiesced"] is True
