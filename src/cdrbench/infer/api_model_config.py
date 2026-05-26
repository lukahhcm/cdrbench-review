from __future__ import annotations

from dataclasses import dataclass


OPENAI_COMPAT_BASE_URL = 'https://api.openai.com/v1'
DEFAULT_COMPAT_MAX_TOKENS = 16384


@dataclass(frozen=True)
class ApiModelConfig:
    model_name: str
    endpoint: str
    input_field: str = 'messages'
    stream: bool = True
    top_level_system: bool = False
    need_max_tokens: bool = False
    default_max_tokens: int = DEFAULT_COMPAT_MAX_TOKENS
    aliases: tuple[str, ...] = ()


# The anonymous release keeps the inference backend provider-agnostic. Add
# entries here only if a deployment needs a non-chat-completions payload shape.
API_MODEL_CONFIGS: tuple[ApiModelConfig, ...] = ()


_MODEL_LOOKUP: dict[str, ApiModelConfig] = {}
for _cfg in API_MODEL_CONFIGS:
    for _name in (_cfg.model_name, *_cfg.aliases):
        _MODEL_LOOKUP[_name.strip().casefold()] = _cfg


def get_api_model_config(model_name: str | None) -> ApiModelConfig | None:
    if not model_name:
        return None
    return _MODEL_LOOKUP.get(model_name.strip().casefold())


def resolve_api_model_name(model_name: str | None, *, default: str) -> str:
    candidate = (model_name or default).strip()
    cfg = get_api_model_config(candidate)
    return cfg.model_name if cfg is not None else candidate


def default_base_url_for_model(model_name: str | None) -> str | None:
    cfg = get_api_model_config(model_name)
    if cfg is None:
        return None
    return OPENAI_COMPAT_BASE_URL
