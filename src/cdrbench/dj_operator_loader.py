from __future__ import annotations

import importlib
import importlib.util
import json
import os
import string
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import regex as re

REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve_data_juicer_repo_root() -> Path:
    env_candidates = [
        os.environ.get('CDRBENCH_DATA_JUICER_ROOT'),
        os.environ.get('ICDRBENCH_DATA_JUICER_ROOT'),
        os.environ.get('DATA_JUICER_ROOT'),
    ]
    candidates = [Path(value).expanduser() for value in env_candidates if value]
    candidates.append(REPO_ROOT / 'data-juicer')
    for candidate in candidates:
        if (candidate / 'data_juicer').exists():
            return candidate
    joined = ', '.join(str(path) for path in candidates)
    raise FileNotFoundError(
        'Unable to locate a Data-Juicer checkout. Set CDRBENCH_DATA_JUICER_ROOT, '
        f'ICDRBENCH_DATA_JUICER_ROOT, or DATA_JUICER_ROOT to the Data-Juicer repo root. Tried: {joined}'
    )


DJ_REPO_ROOT = _resolve_data_juicer_repo_root()
DJ_SOURCE = DJ_REPO_ROOT / 'data_juicer'
DJ_REF_ASSETS = DJ_REPO_ROOT / 'data' / 'src' / 'refs'
LOCAL_REF_ASSETS = REPO_ROOT / 'data' / 'src' / 'refs'


@dataclass(frozen=True)
class OperatorSpec:
    module_name: str
    file_path: Path
    kind: str


def _operator_spec(op_name: str, kind: str) -> OperatorSpec:
    return OperatorSpec(
        module_name=f'data_juicer.ops.{kind}.{op_name}',
        file_path=DJ_SOURCE / 'ops' / kind / f'{op_name}.py',
        kind=kind,
    )


OPERATOR_KINDS: Dict[str, str] = {
    # Pure-text, single-text-input operators (no model inference required).
    'clean_email_mapper': 'mapper',
    'clean_html_mapper': 'mapper',
    'clean_ip_mapper': 'mapper',
    'clean_path_mapper': 'mapper',
    'clean_phone_mapper': 'mapper',
    'clean_id_card_mapper': 'mapper',
    'clean_secret_mapper': 'mapper',
    'clean_channel_id_mapper': 'mapper',
    'clean_jwt_mapper': 'mapper',
    'clean_pem_mapper': 'mapper',
    'clean_mac_mapper': 'mapper',
    'clean_links_mapper': 'mapper',
    'clean_copyright_mapper': 'mapper',
    'expand_macro_mapper': 'mapper',
    'extract_tables_from_html_mapper': 'mapper',
    'fix_unicode_mapper': 'mapper',
    'punctuation_normalization_mapper': 'mapper',
    'remove_bibliography_mapper': 'mapper',
    'remove_comments_mapper': 'mapper',
    'remove_header_mapper': 'mapper',
    'remove_long_words_mapper': 'mapper',
    'remove_repeat_sentences_mapper': 'mapper',
    'remove_specific_chars_mapper': 'mapper',
    'remove_table_text_mapper': 'mapper',
    'remove_words_with_incorrect_substrings_mapper': 'mapper',
    'whitespace_normalization_mapper': 'mapper',
    'alphanumeric_filter': 'filter',
    'average_line_length_filter': 'filter',
    'character_repetition_filter': 'filter',
    'flagged_words_filter': 'filter',
    'maximum_line_length_filter': 'filter',
    'special_characters_filter': 'filter',
    'stopwords_filter': 'filter',
    'text_length_filter': 'filter',
    'word_repetition_filter': 'filter',
    'words_num_filter': 'filter',
}

OPERATOR_SPECS: Dict[str, OperatorSpec] = {
    op_name: _operator_spec(op_name, kind) for op_name, kind in OPERATOR_KINDS.items()
}


def _discover_operator_spec(op_name: str) -> OperatorSpec:
    mapper_path = DJ_SOURCE / 'ops' / 'mapper' / f'{op_name}.py'
    if mapper_path.exists():
        return _operator_spec(op_name, 'mapper')

    filter_path = DJ_SOURCE / 'ops' / 'filter' / f'{op_name}.py'
    if filter_path.exists():
        return _operator_spec(op_name, 'filter')

    raise KeyError(
        f'Unknown operator: {op_name}. Not found in OPERATOR_KINDS and no source file under '
        f"{DJ_SOURCE / 'ops' / 'mapper'} or {DJ_SOURCE / 'ops' / 'filter'}."
    )


def _resolve_operator_spec(op_name: str) -> OperatorSpec:
    spec = OPERATOR_SPECS.get(op_name)
    if spec is None:
        spec = _discover_operator_spec(op_name)
        OPERATOR_SPECS[op_name] = spec
        OPERATOR_KINDS[op_name] = spec.kind
    return spec

_MODEL_CACHE: Dict[str, Any] = {}


class Registry:
    def __init__(self, name: str):
        self.name = name
        self.modules: Dict[str, Any] = {}

    def register_module(self, name: str):
        def decorator(obj):
            self.modules[name] = obj
            return obj

        return decorator


class Fields:
    stats = '__dj__stats__'
    meta = '__dj__meta__'
    batch_meta = '__dj__batch_meta__'
    context = '__dj__context__'
    suffix = '__dj__suffix__'
    text_tags = '__dj__text_tags__'
    source_file = '__dj__source_file__'


class MetaKeys:
    html_tables = 'html_tables'


class StatsKeys:
    text_len = 'text_len'
    avg_line_length = 'avg_line_length'
    max_line_length = 'max_line_length'
    special_char_ratio = 'special_char_ratio'
    alnum_ratio = 'alnum_ratio'
    alpha_token_ratio = 'alpha_token_ratio'
    num_words = 'num_words'
    num_token = 'num_token'
    stopwords_ratio = 'stopwords_ratio'
    flagged_words_ratio = 'flagged_words_ratio'
    char_rep_ratio = 'char_rep_ratio'
    word_rep_ratio = 'word_rep_ratio'
    lang = 'lang'
    lang_score = 'lang_score'


class InterVars:
    lines = '__dj__lines'
    words = '__dj__words'
    refined_words = '__dj__refined_words'


class LazyLoader:
    def __init__(self, module_name: str, _package_name: str | None = None):
        self.module_name = module_name
        self._module = None

    def _load(self):
        if self._module is None:
            self._module = importlib.import_module(self.module_name)
        return self._module

    def __getattr__(self, item: str):
        return getattr(self._load(), item)


class OP:
    _batched_op = True
    accelerator = 'cpu'

    def __init__(self, *args, **kwargs):
        self.text_key = kwargs.get('text_key', 'text')
        self.skip_op_error = kwargs.get('skip_op_error', False)
        self.batch_size = kwargs.get('batch_size', 1000)
        self._name = getattr(self, '_name', self.__class__.__name__)

    def is_batched_op(self) -> bool:
        return bool(getattr(self, '_batched_op', False))

    @staticmethod
    def remove_extra_parameters(local_vars: Dict[str, Any]) -> Dict[str, Any]:
        return {
            key: value
            for key, value in local_vars.items()
            if key not in {'self', 'args', 'kwargs', '__class__'}
        }


class Mapper(OP):
    pass


class Filter(OP):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_closed_interval = kwargs.get('min_closed_interval', True)
        self.max_closed_interval = kwargs.get('max_closed_interval', True)
        self.reversed_range = kwargs.get('reversed_range', False)
        if self.reversed_range:
            self.min_closed_interval = not self.min_closed_interval
            self.max_closed_interval = not self.max_closed_interval

    def get_keep_boolean(self, val, min_val=None, max_val=None):
        res = True
        if min_val is not None:
            res = res and (val >= min_val if self.min_closed_interval else val > min_val)
        if max_val is not None:
            res = res and (val <= max_val if self.max_closed_interval else val < max_val)
        if self.reversed_range:
            res = not res
        return res


OPERATORS = Registry('Operators')
NON_STATS_FILTERS = Registry('NonStatsFilters')
TAGGING_OPS = Registry('TaggingOps')
INTER_LINES = Registry(InterVars.lines)
INTER_WORDS = Registry(InterVars.words)

VARIOUS_WHITESPACES = {
    ' ',
    '\t',
    ' ',
    ' ',
    ' ',
    ' ',
    ' ',
    ' ',
    ' ',
    ' ',
    ' ',
    ' ',
    ' ',
    ' ',
    ' ',
    ' ',
    '　',
    '​',
    '‌',
    '‍',
    '⁠',
    '￼',
    '\x84',
}
SPECIAL_CHARACTERS = set(string.punctuation + string.digits + string.whitespace)
SPECIAL_CHARACTERS.update('◆●■►▼▲▴∆▻▷❖♡□')


def strip(document: str, strip_characters: set[str]) -> str:
    if not document:
        return document
    beg_ind = 0
    end_ind = len(document)
    for char in document:
        if char in strip_characters:
            beg_ind += 1
        else:
            break
    for i in range(1, len(document) + 1):
        if document[-i] in strip_characters:
            end_ind -= 1
        else:
            break
    return document[beg_ind:end_ind]


def split_on_whitespace(document: str, new_line: bool = False, tab: bool = False) -> list[str]:
    sep = [' '] + new_line * ['\n'] + tab * ['\t']
    split_document = re.split('|'.join(sep), document)
    return [word for word in split_document if word]


def split_on_newline_tab_whitespace(document: str) -> list[list[list[str]]]:
    sentences = document.split('\n')
    sentences = [sentence.split('\t') for sentence in sentences]
    return [[split_on_whitespace(subsentence) for subsentence in sentence] for sentence in sentences]


def merge_on_whitespace_tab_newline(sentences: list[list[list[str]]]) -> str:
    lines = ['\t'.join(' '.join(subsentence) for subsentence in sentence if subsentence) for sentence in sentences]
    lines = [line for line in lines if line]
    return '\n'.join(lines) if lines else ''


def words_augmentation(words: list[str], group_size: int, join_char: str) -> list[str]:
    return [join_char.join(words[i : i + group_size]) for i in range(len(words) - group_size + 1)]


def get_words_from_document(document: str, token_func=None, new_line: bool = True, tab: bool = True) -> list[str]:
    if token_func:
        return token_func(document)
    return split_on_whitespace(document, new_line, tab)


def words_refinement(
    words: list[str],
    lower_case: bool = False,
    strip_chars: set[str] | None = None,
    use_words_aug: bool = False,
    words_aug_group_sizes: list[int] | None = None,
    words_aug_join_char: str = '',
) -> list[str]:
    group_sizes = words_aug_group_sizes or [2]
    refined = words
    if lower_case:
        refined = [word.lower() for word in refined]
    if strip_chars:
        refined = [strip(word, strip_chars) for word in refined]
        refined = [word for word in refined if word]
    if use_words_aug:
        augmented = [word for size in group_sizes for word in words_augmentation(refined, size, words_aug_join_char)]
        refined = refined + augmented
    return refined


def get_sentences_from_document(document: str, model_func=None) -> str:
    if model_func:
        return '\n'.join(model_func(document))
    return '\n'.join(document.splitlines())


def _ensure_module(name: str, package: bool = False) -> types.ModuleType:
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        if package:
            module.__path__ = []
        sys.modules[name] = module
    return module


def _load_words_asset(words_dir: str, words_type: str) -> Dict[str, list[str]]:
    words_dict: Dict[str, list[str]] = {}
    search_dirs = [Path(words_dir), LOCAL_REF_ASSETS, DJ_REF_ASSETS]
    for base_dir in search_dirs:
        if not base_dir.exists():
            continue
        for filename in base_dir.glob(f'*{words_type}*.json'):
            with open(filename, 'r', encoding='utf-8') as f:
                loaded_words = json.load(f)
            for key, values in loaded_words.items():
                words_dict.setdefault(key, [])
                words_dict[key].extend(values)
        if words_dict:
            break
    if not words_dict:
        raise FileNotFoundError(f'Unable to find local asset file for {words_type}')
    return words_dict


def _prepare_model(model_type: str, **kwargs):
    return {'model_type': model_type, **kwargs}


def _get_model(model_key, rank=None):
    if not model_key:
        return None
    model_type = model_key.get('model_type')
    cache_key = json.dumps(model_key, sort_keys=True, ensure_ascii=False)
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    if model_type == 'huggingface':
        if importlib.util.find_spec('transformers') is None:
            pretrained_name = model_key.get('pretrained_model_name_or_path') or '<hf_tokenizer>'
            raise RuntimeError(
                'Strict Data-Juicer execution requires the `transformers` package for '
                f'huggingface tokenizer loading (needed by tokenizer={pretrained_name}). '
                'Install it with `pip install transformers sentencepiece`.'
            )
        from transformers import AutoTokenizer  # type: ignore

        pretrained_name = model_key.get('pretrained_model_name_or_path')
        if not pretrained_name:
            raise RuntimeError('Missing `pretrained_model_name_or_path` for huggingface tokenizer loading.')
        tokenizer = AutoTokenizer.from_pretrained(str(pretrained_name))
        _MODEL_CACHE[cache_key] = tokenizer
        return tokenizer

    if model_type == 'sentencepiece':
        if importlib.util.find_spec('sentencepiece') is None:
            raise RuntimeError(
                'Strict Data-Juicer execution requires the `sentencepiece` package for '
                'sentencepiece tokenizer loading. Install it with `pip install sentencepiece`.'
            )
        import sentencepiece as spm  # type: ignore

        model_path = model_key.get('model_path') or model_key.get('pretrained_model_name_or_path')
        if not model_path:
            raise RuntimeError('Missing `model_path` or `pretrained_model_name_or_path` for sentencepiece loading.')
        processor = spm.SentencePieceProcessor()
        loaded = processor.Load(str(model_path))
        if not loaded:
            raise RuntimeError(f'Failed to load sentencepiece model from {model_path}.')
        _MODEL_CACHE[cache_key] = processor
        return processor

    return None


def _patch_nltk_pickle_security() -> None:
    return None


def install_shims() -> None:
    data_juicer_mod = _ensure_module('data_juicer', package=True)
    utils_mod = _ensure_module('data_juicer.utils', package=True)
    ops_mod = _ensure_module('data_juicer.ops', package=True)
    mapper_mod = _ensure_module('data_juicer.ops.mapper', package=True)
    filter_mod = _ensure_module('data_juicer.ops.filter', package=True)
    common_pkg_mod = _ensure_module('data_juicer.ops.common', package=True)
    data_juicer_mod.__path__ = [str(DJ_SOURCE)]
    utils_mod.__path__ = [str(DJ_SOURCE / 'utils')]
    ops_mod.__path__ = [str(DJ_SOURCE / 'ops')]
    mapper_mod.__path__ = [str(DJ_SOURCE / 'ops' / 'mapper')]
    filter_mod.__path__ = [str(DJ_SOURCE / 'ops' / 'filter')]
    common_pkg_mod.__path__ = [str(DJ_SOURCE / 'ops' / 'common')]

    constant_mod = _ensure_module('data_juicer.utils.constant')
    constant_mod.Fields = Fields
    constant_mod.MetaKeys = MetaKeys
    constant_mod.StatsKeys = StatsKeys
    constant_mod.InterVars = InterVars

    lazy_mod = _ensure_module('data_juicer.utils.lazy_loader')
    lazy_mod.LazyLoader = LazyLoader

    asset_mod = _ensure_module('data_juicer.utils.asset_utils')
    asset_mod.ASSET_DIR = str(DJ_REF_ASSETS)
    asset_mod.load_words_asset = _load_words_asset

    model_mod = _ensure_module('data_juicer.utils.model_utils')
    model_mod.prepare_model = _prepare_model
    model_mod.get_model = _get_model

    nltk_mod = _ensure_module('data_juicer.utils.nltk_utils')
    nltk_mod.patch_nltk_pickle_security = _patch_nltk_pickle_security

    base_op_mod = _ensure_module('data_juicer.ops.base_op')
    base_op_mod.OPERATORS = OPERATORS
    base_op_mod.NON_STATS_FILTERS = NON_STATS_FILTERS
    base_op_mod.TAGGING_OPS = TAGGING_OPS
    base_op_mod.Mapper = Mapper
    base_op_mod.Filter = Filter
    base_op_mod.OP = OP

    common_mod = _ensure_module('data_juicer.ops.common')
    helper_mod = _ensure_module('data_juicer.ops.common.helper_func')
    special_mod = _ensure_module('data_juicer.ops.common.special_characters')

    for module in (common_mod, helper_mod):
        module.SPECIAL_CHARACTERS = SPECIAL_CHARACTERS
        module.strip = strip
        module.split_on_whitespace = split_on_whitespace
        module.split_on_newline_tab_whitespace = split_on_newline_tab_whitespace
        module.merge_on_whitespace_tab_newline = merge_on_whitespace_tab_newline
        module.get_words_from_document = get_words_from_document
        module.words_refinement = words_refinement
        module.get_sentences_from_document = get_sentences_from_document

    special_mod.VARIOUS_WHITESPACES = VARIOUS_WHITESPACES
    special_mod.SPECIAL_CHARACTERS = SPECIAL_CHARACTERS

    op_fusion_mod = _ensure_module('data_juicer.ops.op_fusion')
    op_fusion_mod.INTER_LINES = INTER_LINES
    op_fusion_mod.INTER_WORDS = INTER_WORDS


def load_operator_module(op_name: str):
    install_shims()
    spec = _resolve_operator_spec(op_name)
    if spec.module_name in sys.modules:
        return sys.modules[spec.module_name]
    if not spec.file_path.exists():
        raise FileNotFoundError(f'Operator source not found for {op_name}: {spec.file_path}')
    module_spec = importlib.util.spec_from_file_location(spec.module_name, spec.file_path)
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError(f'Unable to load operator module for {op_name}')
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[spec.module_name] = module
    module_spec.loader.exec_module(module)
    return module


def _ensure_exact_operator_dependencies(op_name: str, kwargs: dict[str, Any]) -> None:
    if op_name == 'fix_unicode_mapper' and importlib.util.find_spec('ftfy') is None:
        raise RuntimeError(
            'Strict Data-Juicer execution for `fix_unicode_mapper` requires the `ftfy` package. '
            'Install it with `pip install ftfy` and retry.'
        )

    if op_name == 'token_num_filter':
        if importlib.util.find_spec('transformers') is None:
            tokenizer_name = kwargs.get('hf_tokenizer') or 'EleutherAI/pythia-6.9b-deduped'
            raise RuntimeError(
                'Strict Data-Juicer execution for `token_num_filter` requires a real Hugging Face tokenizer. '
                f'Install dependencies with `pip install transformers sentencepiece` and ensure tokenizer '
                f'`{tokenizer_name}` is downloadable or cached locally.'
            )


def create_operator(op_name: str, **kwargs):
    _ensure_exact_operator_dependencies(op_name, kwargs)
    load_operator_module(op_name)
    if op_name not in OPERATORS.modules:
        raise RuntimeError(
            f'Operator module loaded but did not register {op_name}. '
            f'Check the @OPERATORS.register_module name in {_resolve_operator_spec(op_name).file_path}.'
        )
    cls = OPERATORS.modules[op_name]
    return cls(**kwargs)


def get_operator_kind(op_name: str) -> str:
    return _resolve_operator_spec(op_name).kind


def get_operator_execution_mode(op_name: str) -> str:
    return 'exact'
