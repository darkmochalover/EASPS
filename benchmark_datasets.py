from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

import numpy as np

from countdown_dataset import (
    DEFAULT_PATH as COUNTDOWN_DEFAULT_PATH,
    build_item as countdown_build_item,
    build_prompt as countdown_build_prompt,
    grade_answer as countdown_grade,
    load_countdown_dataset,
    parse_answer as countdown_parse,
)
from grader_utils.he_grader import extract_code as he_extract_code
from grader_utils.he_execute import check_correctness as he_check_correctness
from grader_utils.parse_utils import parse_answer as parse_boxed_answer
from model_utils import CHAT_MODELS


ROOT = Path(__file__).resolve().parent
MEDQA_LETTERS = ("A", "B", "C", "D")
MEDQA_DATASET_ROOT = "data/medQA/questions/US/4_options"
MEDQA_DATASET_SPLIT = "test"
MEDQA_DEFAULT_PATH = f"{MEDQA_DATASET_ROOT}/phrases_no_exclude_test.jsonl"
MEDQA_QUERY_TEMPLATE = (
    "Answer the following medical multiple-choice question. The last line of "
    "your response should be of the following format: '\\boxed{{LETTER}}' "
    "(without quotes) where LETTER is one of ABCD (e.g. '\\boxed{{A}}'). "
    "Think step by step before answering.\n\n"
    "{question}\n\n"
    "A) {choice_a}\n"
    "B) {choice_b}\n"
    "C) {choice_c}\n"
    "D) {choice_d}"
)
HUMANEVAL_INSTRUCTION = (
    "Complete the following Python function. Write the complete function "
    "definition inside a single ```python code block. Do not add tests, "
    "example usage, or explanations."
)
HUMANEVAL_STOP_SEQS = (
    "\nclass ",
    "\ndef ",
    "\n#",
    "\nif ",
    "\nprint(",
    "\nassert ",
    "\n@",
    "\n```",
    '\n"""',
    "\n'''",
)


def resolve_path(path):
    source = Path(path).expanduser()
    return source if source.is_absolute() else ROOT / source


def load_json_dataset(path):
    with resolve_path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl_dataset(path):
    with resolve_path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def format_prompt(question, model, tokenizer, cot=True):
    suffix = (
        " Please reason step by step, and put your final answer within \\boxed{{}}."
        if cot
        else " Put your final answer within \\boxed{{}}."
    )
    content = "Can you solve the following math problem? " + question + suffix
    if model in CHAT_MODELS:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return content


def math_grade_answer(answer, gold):
    from grader_utils.math_grader import grade_answer

    return grade_answer(answer, gold)


def math_build_item(record, idx, opts):
    return {
        "question": record["prompt"],
        "gold": record["answer"],
        "meta": {},
    }


def math_build_prompt(item, model, tokenizer, cot):
    return format_prompt(item["question"], model, tokenizer, cot)


def math_parse(completion, item, opts):
    return parse_boxed_answer(completion)


def math_grade(answer, item, completion, opts):
    return bool(math_grade_answer(answer, item["gold"]))


def normalize_medqa_answer(answer):
    if answer is None or isinstance(answer, (bool, np.bool_)):
        raise ValueError(f"MedQA answer must be one of A-D or 0-3, got {answer!r}")

    if isinstance(answer, (int, np.integer)):
        index = int(answer)
        if 0 <= index < len(MEDQA_LETTERS):
            return MEDQA_LETTERS[index]
        raise ValueError(f"MedQA answer index must be in 0..3, got {answer!r}")

    if isinstance(answer, (float, np.floating)):
        value = float(answer)
        if math.isfinite(value) and value.is_integer():
            return normalize_medqa_answer(int(value))
        raise ValueError(f"MedQA answer must be one of A-D or 0-3, got {answer!r}")

    text_value = str(answer).strip()
    boxed = parse_boxed_answer(text_value)
    if boxed is not None:
        text_value = boxed.strip()

    text_match = re.fullmatch(r"\\text\s*\{\s*([A-Da-d0-3])\s*\}", text_value)
    if text_match:
        text_value = text_match.group(1)

    text_value = text_value.strip().rstrip(".").strip()
    wrapper_match = re.fullmatch(r"[\(\[\{]?\s*([A-Da-d0-3])\s*[\)\]\}]?", text_value)
    if not wrapper_match:
        raise ValueError(f"MedQA answer must be one of A-D or 0-3, got {answer!r}")

    token = wrapper_match.group(1).upper()
    if token.isdigit():
        return MEDQA_LETTERS[int(token)]
    return token


def medqa_split_from_path(path):
    name = Path(str(path or "")).name.lower()
    for split in ("train", "dev", "test"):
        if re.search(rf"(?:^|_){split}(?:_|\.|$)", name):
            return split
    return MEDQA_DATASET_SPLIT


def _resolve_medqa_path(path):
    source = resolve_path(path or MEDQA_DEFAULT_PATH)
    if source.is_dir():
        source = source / f"phrases_no_exclude_{MEDQA_DATASET_SPLIT}.jsonl"
    if not source.exists():
        raise FileNotFoundError(f"MedQA dataset path does not exist: {source}")
    if not source.is_file():
        raise ValueError(f"MedQA dataset path must be a JSONL file: {source}")
    if source.suffix.lower() != ".jsonl":
        raise ValueError(f"MedQA dataset path must end in .jsonl: {source}")
    return source


def load_medqa_dataset(path):
    source = _resolve_medqa_path(path)
    records = load_jsonl_dataset(source)
    split = medqa_split_from_path(source)
    for idx, record in enumerate(records):
        medqa_build_item(record, idx, {"dataset_split": split})
    return records


def medqa_build_item(record, idx, opts):
    question = str(record.get("question") or "").strip()
    if not question:
        raise ValueError(f"MedQA row {idx} has no question")

    raw_options = record.get("options")
    if not isinstance(raw_options, dict) or set(raw_options) != set(MEDQA_LETTERS):
        raise ValueError(f"MedQA row {idx} must have exactly A-D options")
    choices = []
    for letter in MEDQA_LETTERS:
        value = raw_options[letter]
        choice = "" if value is None else str(value).strip()
        if not choice:
            raise ValueError(f"MedQA row {idx} option {letter} is empty")
        choices.append(choice)

    gold = normalize_medqa_answer(record.get("answer_idx"))
    answer_text = "" if record.get("answer") is None else str(record["answer"]).strip()
    expected_text = choices[MEDQA_LETTERS.index(gold)]
    if answer_text and answer_text != expected_text:
        raise ValueError(f"MedQA row {idx} answer does not match option {gold}")

    return {
        "question": question,
        "gold": gold,
        "meta": {
            "choices": choices,
            "split": opts.get("dataset_split", MEDQA_DATASET_SPLIT),
            "dataset_id": record.get("id", idx),
            "meta_info": record.get("meta_info"),
        },
    }


def medqa_build_prompt(item, model, tokenizer, cot):
    choices = item["meta"]["choices"]
    return MEDQA_QUERY_TEMPLATE.format(
        question=item["question"],
        choice_a=choices[0],
        choice_b=choices[1],
        choice_c=choices[2],
        choice_d=choices[3],
    )


def medqa_parse(completion, item, opts):
    if not completion:
        return None

    boxed = parse_boxed_answer(completion)
    if boxed is not None:
        try:
            return normalize_medqa_answer(boxed)
        except ValueError:
            pass

    explicit = re.findall(
        r"(?i)(?:final\s+answer|correct\s+(?:answer|option|choice)|answer|option|choice)"
        r"\s*(?:is\s*)?(?:[:=]\s*)?(?:\\boxed\s*\{\s*)?"
        r"(?:\\text\s*\{\s*)?[\(\[]?\s*([A-D])\b",
        completion,
    )
    if explicit:
        return explicit[-1].upper()

    for line in reversed(completion.splitlines()):
        final_line = re.fullmatch(
            r"(?i)\s*(?:(?:final\s+)?answer\s*(?:is|:|=)\s*)?"
            r"(?:\\boxed\s*\{\s*)?(?:\\text\s*\{\s*)?"
            r"[\(\[]?\s*([A-D])\s*[\)\]\}]?\s*[.]?\s*\}?\s*",
            line,
        )
        if final_line:
            return final_line.group(1).upper()

    standalone = re.findall(r"(?<![A-Za-z])([A-D])(?![A-Za-z])", completion)
    return standalone[-1] if standalone else None


def medqa_grade(answer, item, completion, opts):
    try:
        return normalize_medqa_answer(answer) == normalize_medqa_answer(item["gold"])
    except ValueError:
        return False


def humaneval_build_item(record, idx, opts):
    return {
        "question": record["prompt"],
        "gold": record["task_id"],
        "meta": {"problem": record, "entry_point": record["entry_point"]},
    }


def humaneval_build_prompt(item, model, tokenizer, cot):
    stub = item["question"]
    if model in CHAT_MODELS:
        content = f"{HUMANEVAL_INSTRUCTION}\n\n```python\n{stub}```\n"
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
        )

    return stub


def humaneval_truncate(completion):
    cut = len(completion)
    for stop in HUMANEVAL_STOP_SEQS:
        pos = completion.find(stop)
        if pos != -1:
            cut = min(cut, pos)
    return completion[:cut]


def humaneval_parse(completion, item, opts):
    if opts.get("model") in CHAT_MODELS:
        return he_extract_code(completion, item["meta"]["entry_point"])
    return humaneval_truncate(completion)


def humaneval_grade(answer, item, completion, opts):
    if not answer:
        return False
    try:
        result = he_check_correctness(
            item["meta"]["problem"], answer, opts["he_timeout"], 0
        )
    except Exception as exc:
        print(f"[humaneval] check_correctness failed: {exc}", flush=True)
        return False
    return bool(result["passed"])


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    path: str
    load: Callable
    build_item: Callable
    build_prompt: Callable
    parse: Callable
    grade: Callable
    stop_on_boxed: bool = True
    result_prefix: str = "math"
    source: str = ""
    config: str = ""
    revision: str = ""
    split: str = ""


DATASET_SPECS = {
    "MATH": DatasetSpec(
        name="MATH",
        path="data/MATH500.json",
        load=load_json_dataset,
        build_item=math_build_item,
        build_prompt=math_build_prompt,
        parse=math_parse,
        grade=math_grade,
    ),
    "COUNTDOWN": DatasetSpec(
        name="COUNTDOWN",
        path=COUNTDOWN_DEFAULT_PATH,
        load=load_countdown_dataset,
        build_item=countdown_build_item,
        build_prompt=countdown_build_prompt,
        parse=countdown_parse,
        grade=countdown_grade,
        result_prefix="countdown",
        source=COUNTDOWN_DEFAULT_PATH,
        split="test",
    ),
    "MEDQA": DatasetSpec(
        name="MedQA",
        path=MEDQA_DEFAULT_PATH,
        load=load_medqa_dataset,
        build_item=medqa_build_item,
        build_prompt=medqa_build_prompt,
        parse=medqa_parse,
        grade=medqa_grade,
        result_prefix="medqa",
        source=MEDQA_DATASET_ROOT,
        split=MEDQA_DATASET_SPLIT,
    ),
    "HUMANEVAL": DatasetSpec(
        name="HUMANEVAL",
        path="data/HumanEval.jsonl",
        load=load_jsonl_dataset,
        build_item=humaneval_build_item,
        build_prompt=humaneval_build_prompt,
        parse=humaneval_parse,
        grade=humaneval_grade,
        stop_on_boxed=False,
        result_prefix="humaneval",
    ),
}


def get_dataset_spec(dataset_name, path_override=""):
    key = str(dataset_name).strip().upper()
    if key not in DATASET_SPECS:
        raise ValueError(
            f"unknown --dataset '{dataset_name}'. supported: {', '.join(DATASET_SPECS)}"
        )
    spec = DATASET_SPECS[key]
    if path_override:
        path_override = str(path_override)
        loader = spec.load
        if key in {"MATH", "HUMANEVAL"} and path_override.endswith(".jsonl"):
            loader = load_jsonl_dataset
        spec = replace(
            spec,
            path=path_override,
            load=loader,
            source=path_override,
            split=medqa_split_from_path(path_override)
            if key == "MEDQA"
            else spec.split,
        )
    path = _resolve_medqa_path(spec.path) if key == "MEDQA" else resolve_path(spec.path)
    return replace(spec, path=str(path))
