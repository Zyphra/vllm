"""RSA parameter and server configuration."""

import argparse
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

Selection = Literal["auto", "majority", "final_agg", "sample"]


class RSAParams(BaseModel):
    """Tunable RSA parameters.

    Defaults match the ZAYA1-8B report's Markovian RSA configuration
    (N=16, K=4, T=2, tau=4096, beta=40000). Set ``tail_tokens=0`` for
    full-trace generalized RSA.
    """

    enabled: bool = True
    n: int = Field(default=16, ge=1, description="population size N")
    k: int = Field(default=4, ge=1, description="aggregation set size K")
    t: int = Field(default=2, ge=1, description="total rounds T")
    tail_tokens: int = Field(
        default=4096, ge=0, description="tau; 0 = full trace (generalized RSA)"
    )
    max_tokens: int = Field(
        default=40000, ge=1, description="beta; per-rollout completion budget"
    )
    temperature: float = Field(default=0.8, ge=0.0)
    selection: Selection = "auto"
    max_concurrency: int = Field(default=16, ge=1)
    request_timeout: float = Field(default=1800.0, gt=0)
    max_retries: int = Field(default=1, ge=0)


def merge_params(defaults: RSAParams, rsa_value) -> RSAParams | None:
    """Merge a request's ``rsa`` extra-body value over server defaults.

    Returns None when the request opts out of RSA (``"rsa": false`` or
    ``{"enabled": false}``). ``"rsa": true`` or absent -> server defaults.
    """
    if rsa_value is None or rsa_value is True:
        merged = defaults
    elif rsa_value is False:
        return None
    elif isinstance(rsa_value, dict):
        merged = defaults.model_copy(
            update=RSAParams(**{**defaults.model_dump(), **rsa_value}).model_dump()
        )
    else:
        raise ValueError(f"invalid 'rsa' value: {rsa_value!r}")
    return merged if merged.enabled else None


@dataclass
class ServerConfig:
    backend_base_url: str = "http://localhost:8000/v1"
    host: str = "0.0.0.0"
    port: int = 8100
    api_key: str = "EMPTY"
    log_level: str = "info"
    defaults: RSAParams = field(default_factory=RSAParams)

    @property
    def backend_root(self) -> str:
        """Backend root URL (without /v1) for /tokenize and /detokenize."""
        return self.backend_base_url.rstrip("/").removesuffix("/v1")


def add_rsa_args(parser: argparse.ArgumentParser) -> None:
    d = RSAParams()
    g = parser.add_argument_group("RSA parameters")
    g.add_argument("--rsa-n", type=int, default=d.n, help="population size N")
    g.add_argument("--rsa-k", type=int, default=d.k, help="aggregation set size K")
    g.add_argument("--rsa-t", type=int, default=d.t, help="total rounds T")
    g.add_argument(
        "--rsa-tail-tokens",
        type=int,
        default=d.tail_tokens,
        help="tau; tail tokens carried into aggregation prompts (0 = full trace)",
    )
    g.add_argument(
        "--rsa-max-tokens",
        type=int,
        default=d.max_tokens,
        help="beta; per-rollout completion budget",
    )
    g.add_argument("--rsa-temperature", type=float, default=d.temperature)
    g.add_argument(
        "--rsa-selection",
        choices=["auto", "majority", "final_agg", "sample"],
        default=d.selection,
    )
    g.add_argument("--rsa-max-concurrency", type=int, default=d.max_concurrency)
    g.add_argument("--rsa-request-timeout", type=float, default=d.request_timeout)
    g.add_argument("--rsa-max-retries", type=int, default=d.max_retries)


def params_from_args(args: argparse.Namespace) -> RSAParams:
    return RSAParams(
        n=args.rsa_n,
        k=args.rsa_k,
        t=args.rsa_t,
        tail_tokens=args.rsa_tail_tokens,
        max_tokens=args.rsa_max_tokens,
        temperature=args.rsa_temperature,
        selection=args.rsa_selection,
        max_concurrency=args.rsa_max_concurrency,
        request_timeout=args.rsa_request_timeout,
        max_retries=args.rsa_max_retries,
    )
