from __future__ import annotations

import math
import re
from collections import Counter


_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_ASIN_PATTERN = re.compile(r"^b0[a-z0-9]{8}$")


def tokenize(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(text.lower())


def cosine_text_similarity(left: str, right: str) -> float:
    left_counts = Counter(tokenize(left))
    right_counts = Counter(tokenize(right))
    if not left_counts or not right_counts:
        return 0.0

    overlap = set(left_counts) & set(right_counts)
    dot = sum(left_counts[token] * right_counts[token] for token in overlap)
    left_norm = math.sqrt(sum(value * value for value in left_counts.values()))
    right_norm = math.sqrt(sum(value * value for value in right_counts.values()))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def action_template(action: str) -> str:
    action = action.strip().lower()
    if action.startswith("search["):
        return "search_query"
    if action.startswith("click["):
        target = action[len("click[") : -1].strip() if action.endswith("]") else action[6:].strip()
        return _webshop_click_template(target)

    parts = action.split()
    if not parts:
        return "unknown_action"
    if parts[0] == "go":
        return "navigate"
    if parts[0] in {"take", "put", "heat", "cool", "clean", "open", "close", "examine", "look"}:
        return parts[0]
    return parts[0]


def compact_counter(counter: Counter[str], limit: int = 5) -> list[dict[str, object]]:
    return [
        {
            "value": value,
            "count": count,
        }
        for value, count in counter.most_common(limit)
    ]


def _webshop_click_template(target: str) -> str:
    if _ASIN_PATTERN.match(target):
        return "click_product"
    if target in {"description", "features", "reviews"}:
        return "inspect_product"
    if target in {"buy now", "buy"}:
        return "purchase"
    if target in {"back to search", "< prev", "prev", "next >", "next", "search"}:
        return "navigate_page"
    if target in {
        "black",
        "white",
        "gray",
        "grey",
        "green",
        "blue",
        "red",
        "orange",
        "yellow",
        "pink",
        "purple",
        "brown",
        "small",
        "medium",
        "large",
        "x-large",
        "xx-large",
        "3x-large",
    }:
        return "select_option"
    return "click_option"
