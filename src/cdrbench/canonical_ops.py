from __future__ import annotations

import html
import re
from collections.abc import Callable
from typing import Any


MapperFn = Callable[[str, dict[str, Any], str], str]


URL_RE = re.compile(
    r"""(?ix)
    (?:
        \b(?:https?|ftp)://[^\s<>()\[\]{}'"`]+
      | \bwww\d{0,3}\.[^\s<>()\[\]{}'"`]+
      | \b[a-z0-9][a-z0-9.-]+\.[a-z]{2,24}/[^\s<>()\[\]{}'"`]*
      | \bmailto:[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}
      | \b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}
    )
    """
)

HTML_COMMENT_RE = re.compile(r'<!--.*?-->', re.DOTALL)
HTML_TAG_RE = re.compile(r'<[^>\n]+>')

WINDOWS_PATH_RE = re.compile(r'(?<!\w)(?:[A-Za-z]:\\|\\\\)[^\s<>"|?*]+')
UNIX_PATH_RE = re.compile(r'(?<!\w)/(?:[\w.-]+/)+[\w.-]*')

PHONE_RE = re.compile(
    r"""(?x)
    (?<![\w])
    (?:\+?\d{1,3}[\s.-]?)?
    (?:\(?\d{2,4}\)?[\s.-]?)?
    \d{3,4}[\s.-]?\d{4}
    (?![\w])
    """
)


def has_canonical_mapper(op_name: str) -> bool:
    return op_name in CANONICAL_MAPPERS


def apply_canonical_mapper(op_name: str, text: str, params: dict[str, Any] | None = None, suffix: str = '') -> str:
    try:
        mapper = CANONICAL_MAPPERS[op_name]
    except KeyError as exc:
        raise KeyError(f'No canonical mapper registered for {op_name}') from exc
    return mapper(text, dict(params or {}), suffix)


def clean_links_mapper(text: str, params: dict[str, Any], suffix: str) -> str:
    """Remove explicit web/email links without treating wiki namespaces as URLs."""

    repl = str(params.get('repl', ''))
    return URL_RE.sub(repl, text)


def clean_path_mapper(text: str, params: dict[str, Any], suffix: str) -> str:
    """Remove explicit filesystem paths, not bare filenames or LaTeX figure names."""

    repl = str(params.get('repl', ''))
    text = WINDOWS_PATH_RE.sub(repl, text)
    text = UNIX_PATH_RE.sub(repl, text)
    return text


def clean_phone_mapper(text: str, params: dict[str, Any], suffix: str) -> str:
    """Remove phone-number-like spans while avoiding arbitrary short numbers."""

    repl = str(params.get('repl', ''))
    return PHONE_RE.sub(repl, text)


def clean_html_mapper(text: str, params: dict[str, Any], suffix: str) -> str:
    """Remove HTML tags/comments and decode entities without converting wiki markup."""

    repl = str(params.get('repl', ''))
    text = HTML_COMMENT_RE.sub(repl, text)
    text = HTML_TAG_RE.sub(repl, text)
    return html.unescape(text)


def remove_comments_mapper(text: str, params: dict[str, Any], suffix: str) -> str:
    """Remove comments for the apparent document type, avoiding TeX comment rules on wiki text."""

    inline = bool(params.get('inline', True))
    multiline = bool(params.get('multiline', True))
    lowered = text.lower()
    looks_tex = any(
        marker in text
        for marker in (
            '\\documentclass',
            '\\begin{',
            '\\end{',
            '\\section',
            '\\subsection',
            '\\usepackage',
            '\\title{',
            '\\author{',
            '\\maketitle',
            '\\bibliography',
        )
    )

    if '<!--' in text:
        text = HTML_COMMENT_RE.sub('', text)

    if looks_tex:
        if inline:
            text = re.sub(r'(?m)(?<!\\)%.*$', '', text)
        if multiline:
            text = re.sub(r'(?m)^%.*\n?', '', text)
    elif lowered.strip().startswith(('#', '//')):
        text = re.sub(r'(?m)^\s*(?:#|//).*\n?', '', text)
    return text


def remove_specific_chars_mapper(text: str, params: dict[str, Any], suffix: str) -> str:
    chars = params.get('chars_to_remove', '◆●■►▼▲▴∆▻▷❖♡□')
    if isinstance(chars, str):
        chars_to_remove = set(chars)
    else:
        chars_to_remove = {str(item) for item in chars}
    if not chars_to_remove:
        return text
    return ''.join(ch for ch in text if ch not in chars_to_remove)


def remove_long_words_mapper(text: str, params: dict[str, Any], suffix: str) -> str:
    min_len = int(params.get('min_len', 1) or 1)
    max_len = int(params.get('max_len', 2**63 - 1) or (2**63 - 1))

    def replace(match: re.Match[str]) -> str:
        token = match.group(0)
        stripped = token.strip('''!"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~''')
        return token if min_len <= len(stripped) <= max_len else ''

    return re.sub(r'\S+', replace, text)


def remove_words_with_incorrect_substrings_mapper(text: str, params: dict[str, Any], suffix: str) -> str:
    substrings = params.get('substrings') or ['http', 'www', '.com', 'href', '//']
    lowered_substrings = [str(item).lower() for item in substrings]

    def replace(match: re.Match[str]) -> str:
        token = match.group(0)
        lowered = token.strip('''!"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~''').lower()
        return '' if any(substring in lowered for substring in lowered_substrings) else token

    return re.sub(r'\S+', replace, text)


def remove_repeat_sentences_mapper(text: str, params: dict[str, Any], suffix: str) -> str:
    lowercase = bool(params.get('lowercase', False))
    min_len = int(params.get('min_repeat_sentence_length', 2) or 2)
    ignore_special = bool(params.get('ignore_special_character', True))
    sentence_re = re.compile(r'[^.!?\n]+[.!?]?|\n+')
    seen: set[str] = set()
    output: list[str] = []
    for match in sentence_re.finditer(text):
        segment = match.group(0)
        if not segment.strip() or segment.startswith('\n'):
            output.append(segment)
            continue
        key = segment.strip()
        if lowercase:
            key = key.lower()
        if ignore_special:
            key = re.sub(r'[^A-Za-z0-9\u4e00-\u9fff]+', '', key)
        if len(key) < min_len or key not in seen:
            output.append(segment)
            seen.add(key)
    return ''.join(output)


CANONICAL_MAPPERS: dict[str, MapperFn] = {
    'clean_links_mapper': clean_links_mapper,
    'clean_path_mapper': clean_path_mapper,
    'clean_phone_mapper': clean_phone_mapper,
    'clean_html_mapper': clean_html_mapper,
    'remove_comments_mapper': remove_comments_mapper,
    'remove_specific_chars_mapper': remove_specific_chars_mapper,
    'remove_long_words_mapper': remove_long_words_mapper,
    'remove_words_with_incorrect_substrings_mapper': remove_words_with_incorrect_substrings_mapper,
}
