from typing import List, Optional, Union
from markdown_it import MarkdownIt
import json
import json5


def extract_json_like_blocks(
    md_text: str,
    prefer_json5: bool = True
) -> List[Union[dict, list]]:
    """
    Extract all json/json5 code blocks from Markdown and parse them,
    compatible with cases where the entire text is a JSON/JSON5 text.
    """
    md = MarkdownIt()
    tokens = md.parse(md_text)
    results: List[Union[dict, list]] = []

    # Common parsing function: priority determined by prefer_json5
    def try_parse(txt: str) -> Optional[Union[dict, list]]:
        txt = txt.strip()
        if not txt:
            return None

        parsers = (
            [json5.loads, json.loads]
            if prefer_json5
            else [json.loads, json5.loads]
        )

        for parser in parsers:
            try:
                return parser(txt)
            except Exception:
                continue
        return None

    # 1. First look in fenced code blocks
    for token in tokens:
        if token.type != "fence":
            continue

        info = (token.info or "").strip().lower()  # e.g., "json", "json5", "js json5"
        content = token.content

        # Check if this code block is marked as json/json5
        is_json_block = (
            "json" in info or "json5" in info or info == ""
        )

        if not is_json_block:
            continue

        obj = try_parse(content)
        if obj is not None:
            results.append(obj)

    # 2. If nothing was parsed from fenced blocks, try treating the entire text as JSON/JSON5
    if not results:
        obj = try_parse(md_text)
        if obj is not None:
            results.append(obj)

    return results


def load_first_json_like(
    md_text: str,
    prefer_json5: bool = True
) -> Union[dict, list, None]:
    """
    Return the first successfully parsed JSON/JSON5 code block from Markdown,
    or the object when the entire text itself is JSON/JSON5.
    """
    blocks = extract_json_like_blocks(md_text, prefer_json5=prefer_json5)
    return blocks[0] if blocks else None