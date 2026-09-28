"""Countdown dataset, prompt, parser, and grader shared by benchmark runners."""

from __future__ import annotations

import ast
import json
import re
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Any

from model_utils import CHAT_MODELS


DEFAULT_PATH = "data/countdown_cd3_test.jsonl"
EXPECTED_ITEMS = 256
PROMPT_PREFIX = (
    "Using only the provided numbers, create an arithmetic expression that "
    "evaluates exactly to the target. Please reason step by step, and put your "
    "final expression within \\boxed{}.\n"
    "Use each number exactly once. You may use +, -, *, and /. Do not include "
    "an equals sign or the target number in the final expression.\n"
    "Numbers: "
)
PROMPT_SUFFIX = "\nRemember to present your final expression within \\boxed{}."

_BOX_START_RE = re.compile(
    r"(?i)(?:\\boxed|\\fbox|(?<![A-Za-z])boxed|(?<![A-Za-z])fbox)\s*\{"
)
_BOXED_SPACE_RE = re.compile(r"(?im)\\boxed\s+([^\r\n]+)")
_ANSWER_TAG_RE = re.compile(r"<answer\b[^>]*>(.*?)(?:</answer>|\Z)", re.I | re.S)
_FINAL_MARKER_RE = re.compile(
    r"(?i)\b(?:final\s+(?:answer|expression|solution)|answer|expression|solution)"
    r"\s*[:=]"
)
_FRAC_RE = re.compile(r"\\(?:dfrac|tfrac|frac)\s*")
_INTEGER_RE = re.compile(r"[+-]?\d+")

_ALLOWED_BINARY_OPERATORS = (ast.Add, ast.Sub, ast.Mult, ast.Div)
_ALLOWED_UNARY_OPERATORS = (ast.UAdd, ast.USub)
_MAX_EXPRESSION_CHARS = 512
_MAX_AST_NODES = 64


def _normalize_integer(value: Any, *, field: str, row_idx: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"Countdown row {row_idx} {field} must be an integer")
    if isinstance(value, int):
        number = value
    else:
        text = str(value if value is not None else "").strip()
        if not _INTEGER_RE.fullmatch(text):
            raise ValueError(
                f"Countdown row {row_idx} {field} must be an integer, got {value!r}"
            )
        number = int(text)
    if not 1 <= number <= 100:
        raise ValueError(
            f"Countdown row {row_idx} {field} must be in 1..100, got {number}"
        )
    return number


def _normalize_numbers(value: Any, *, row_idx: int) -> tuple[int, int, int]:
    if isinstance(value, str):
        raw_numbers = [part.strip() for part in value.split(",")]
    elif isinstance(value, (list, tuple)):
        raw_numbers = list(value)
    else:
        raise ValueError(
            f"Countdown row {row_idx} input/nums must be a comma-separated "
            f"string or list, got {value!r}"
        )
    if len(raw_numbers) != 3:
        raise ValueError(
            f"Countdown row {row_idx} must contain exactly 3 numbers, "
            f"got {raw_numbers!r}"
        )
    numbers = tuple(
        _normalize_integer(number, field=f"number {position}", row_idx=row_idx)
        for position, number in enumerate(raw_numbers)
    )
    return numbers


def normalize_record(record: dict[str, Any], row_idx: int) -> dict[str, str]:
    """Normalize d1's input/output schema and compatible nums/target mirrors."""
    raw_numbers = record.get("input")
    if raw_numbers is None:
        raw_numbers = record.get("nums")
    raw_target = record.get("output")
    if raw_target is None:
        raw_target = record.get("target")

    numbers = _normalize_numbers(raw_numbers, row_idx=row_idx)
    target = _normalize_integer(raw_target, field="target", row_idx=row_idx)
    return {
        "input": ",".join(str(number) for number in numbers),
        "output": str(target),
    }


def _resolve_source(path: str | Path) -> Path:
    source = Path(path or DEFAULT_PATH)
    if source.is_absolute():
        return source
    return Path(__file__).resolve().parent / source


def load_countdown_dataset(
    path: str | Path = DEFAULT_PATH,
) -> list[dict[str, str]]:
    """Load the d1 Countdown cd3 JSONL test set with strict schema validation."""
    source = _resolve_source(path)
    if not source.is_file():
        raise FileNotFoundError(f"Countdown dataset path does not exist: {source}")
    if source.suffix.lower() != ".jsonl":
        raise ValueError(f"Countdown dataset must be a .jsonl file, got: {source}")

    records = []
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw_record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Countdown JSONL line {line_number} is invalid JSON"
                ) from exc
            if not isinstance(raw_record, dict):
                raise ValueError(
                    f"Countdown JSONL line {line_number} must contain an object"
                )
            records.append(normalize_record(raw_record, line_number - 1))

    if not records:
        raise ValueError("Countdown dataset is empty")
    default_source = _resolve_source(DEFAULT_PATH)
    if source.resolve() == default_source.resolve() and len(records) != EXPECTED_ITEMS:
        raise ValueError(
            f"Canonical Countdown test set must contain {EXPECTED_ITEMS} rows, "
            f"got {len(records)}"
        )
    if len({(row["input"], row["output"]) for row in records}) != len(records):
        raise ValueError("Countdown dataset contains duplicate input/target rows")
    return records


def build_item(
    record: dict[str, Any], idx: int, opts: dict[str, Any]
) -> dict[str, Any]:
    del opts
    normalized = normalize_record(record, idx)
    numbers = tuple(int(part) for part in normalized["input"].split(","))
    target = int(normalized["output"])
    return {
        "question": f"Numbers: {list(numbers)}\nTarget: {target}",
        "gold": str(target),
        "meta": {
            "dataset_id": idx,
            "numbers": numbers,
            "target": target,
        },
    }


def build_prompt(item: dict[str, Any], model: str, tokenizer: Any, cot: bool) -> str:
    """Build the fixed zero-shot Countdown boxed-expression prompt."""
    del cot
    numbers = list(item["meta"]["numbers"])
    target = item["meta"]["target"]
    content = f"{PROMPT_PREFIX}{numbers}\nTarget: {target}{PROMPT_SUFFIX}"
    if model in CHAT_MODELS:
        if tokenizer is None:
            raise ValueError(f"tokenizer is required for chat model {model!r}")
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return content


def _balanced_command_contents(text: str) -> list[str]:
    contents = []
    for match in _BOX_START_RE.finditer(text):
        left = match.end() - 1
        depth = 0
        for position in range(left, len(text)):
            character = text[position]
            if character == "{":
                depth += 1
            elif character == "}":
                depth -= 1
                if depth == 0:
                    contents.append(text[left + 1 : position])
                    break
    return contents


def _braced_content(text: str, start: int) -> tuple[str, int] | None:
    position = start
    while position < len(text) and text[position].isspace():
        position += 1
    if position >= len(text) or text[position] != "{":
        return None
    depth = 0
    for end in range(position, len(text)):
        if text[end] == "{":
            depth += 1
        elif text[end] == "}":
            depth -= 1
            if depth == 0:
                return text[position + 1 : end], end + 1
    return None


def _replace_latex_fractions(text: str) -> str:
    for _ in range(16):
        match = _FRAC_RE.search(text)
        if match is None:
            return text
        numerator = _braced_content(text, match.end())
        if numerator is None:
            return text
        denominator = _braced_content(text, numerator[1])
        if denominator is None:
            return text
        replacement = (
            f"(({_replace_latex_fractions(numerator[0])})/"
            f"({_replace_latex_fractions(denominator[0])}))"
        )
        text = text[: match.start()] + replacement + text[denominator[1] :]
    return text


def normalize_expression(value: Any) -> str:
    """Normalize common plain-text and LaTeX arithmetic spellings."""
    expression = str(value if value is not None else "").strip()
    if expression.startswith("$") and expression.endswith("$") and len(expression) >= 2:
        expression = expression[1:-1].strip()
    if expression.startswith(r"\(") and expression.endswith(r"\)"):
        expression = expression[2:-2].strip()
    if expression.startswith(r"\[") and expression.endswith(r"\]"):
        expression = expression[2:-2].strip()

    expression = _replace_latex_fractions(expression)
    replacements = (
        (r"\left", ""),
        (r"\right", ""),
        (r"\times", "*"),
        (r"\cdot", "*"),
        (r"\div", "/"),
        ("×", "*"),
        ("÷", "/"),
        ("−", "-"),
        (r"\qquad", " "),
        (r"\quad", " "),
        (r"\,", " "),
        (r"\;", " "),
        (r"\:", " "),
        (r"\!", ""),
    )
    for old, new in replacements:
        expression = expression.replace(old, new)
    expression = expression.replace("{", "(").replace("}", ")")
    return expression.strip().rstrip(".").strip()


def _expression_tree(expression: str) -> ast.Expression | None:
    if not expression or len(expression) > _MAX_EXPRESSION_CHARS:
        return None
    try:
        tree = ast.parse(expression, mode="eval")
    except (SyntaxError, ValueError):
        return None
    if sum(1 for _ in ast.walk(tree)) > _MAX_AST_NODES:
        return None

    def valid(node: ast.AST) -> bool:
        if isinstance(node, ast.Constant):
            return (
                isinstance(node.value, int)
                and not isinstance(node.value, bool)
                and node.value >= 0
            )
        if isinstance(node, ast.BinOp):
            return (
                isinstance(node.op, _ALLOWED_BINARY_OPERATORS)
                and valid(node.left)
                and valid(node.right)
            )
        if isinstance(node, ast.UnaryOp):
            return isinstance(node.op, _ALLOWED_UNARY_OPERATORS) and valid(node.operand)
        return False

    return tree if valid(tree.body) else None


def _numbers_in_tree(tree: ast.Expression) -> list[int]:
    return [
        int(node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, int)
        and not isinstance(node.value, bool)
    ]


def _evaluate_node(node: ast.AST) -> Fraction:
    if isinstance(node, ast.Constant):
        return Fraction(int(node.value))
    if isinstance(node, ast.UnaryOp):
        operand = _evaluate_node(node.operand)
        return operand if isinstance(node.op, ast.UAdd) else -operand
    if isinstance(node, ast.BinOp):
        left = _evaluate_node(node.left)
        right = _evaluate_node(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if right == 0:
            raise ZeroDivisionError
        return left / right
    raise ValueError("Unsupported expression node")


def _candidate_expression(value: Any) -> str | None:
    expression = normalize_expression(value)
    return expression if _expression_tree(expression) is not None else None


def parse_answer(
    completion: Any, item: dict[str, Any], opts: dict[str, Any]
) -> str | None:
    """Extract the final arithmetic expression with conservative fallbacks."""
    del item, opts
    text = str(completion or "")

    for content in reversed(_balanced_command_contents(text)):
        expression = _candidate_expression(content)
        if expression is not None:
            return expression
    for content in reversed(_BOXED_SPACE_RE.findall(text)):
        expression = _candidate_expression(content)
        if expression is not None:
            return expression
    for content in reversed(_ANSWER_TAG_RE.findall(text)):
        expression = _candidate_expression(content)
        if expression is not None:
            return expression

    for marker in reversed(list(_FINAL_MARKER_RE.finditer(text))):
        tail = text[marker.end() :].strip()
        for candidate in [tail, *tail.splitlines()]:
            expression = _candidate_expression(candidate)
            if expression is not None:
                return expression

    expression = _candidate_expression(text)
    if expression is not None:
        return expression
    for line in reversed(text.splitlines()):
        expression = _candidate_expression(line)
        if expression is not None:
            return expression
    return None


def expression_metrics(answer: Any, item: dict[str, Any]) -> dict[str, Any]:
    """Return auditable syntax, number-usage, value, and correctness fields."""
    numbers = tuple(int(number) for number in item["meta"]["numbers"])
    target = int(item["meta"]["target"])
    expression = normalize_expression(answer)
    tree = _expression_tree(expression)
    if tree is None:
        return {
            "expression": expression or None,
            "numbers": numbers,
            "target": target,
            "valid_syntax": False,
            "uses_all_numbers": False,
            "value": None,
            "correct": False,
        }

    uses_all_numbers = Counter(_numbers_in_tree(tree)) == Counter(numbers)
    try:
        value = _evaluate_node(tree.body)
    except (ZeroDivisionError, ValueError):
        value = None
    value_text = (
        None
        if value is None
        else str(value.numerator)
        if value.denominator == 1
        else f"{value.numerator}/{value.denominator}"
    )
    correct = uses_all_numbers and value == Fraction(target)
    return {
        "expression": expression,
        "numbers": numbers,
        "target": target,
        "valid_syntax": True,
        "uses_all_numbers": uses_all_numbers,
        "value": value_text,
        "correct": bool(correct),
    }


def grade_answer(
    answer: Any,
    item: dict[str, Any],
    completion: Any,
    opts: dict[str, Any],
) -> bool:
    del completion, opts
    return bool(expression_metrics(answer, item)["correct"])
