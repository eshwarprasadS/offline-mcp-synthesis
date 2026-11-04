"""Teacher-only synthesis of tool-calling conversations. No tool execution."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, ValidationError

ROOT = Path(__file__).resolve().parent
GENERATION_FLOW = ROOT / "generation" / "flow.yaml"
REVIEW_FLOW = ROOT / "review" / "flow.yaml"
FILTER_FLOW = ROOT / "filter" / "flow.yaml"
INPUT_DIR = ROOT / "input"
OUTPUT_DIR = ROOT / "output"
DEFAULT_SCHEMA = INPUT_DIR / "tool_schema.json"
DEFAULT_EXAMPLES = INPUT_DIR / "examples.jsonl"
DEFAULT_SEEDS = INPUT_DIR / "seeds.json"
DEFAULT_PROMPT = INPUT_DIR / "system_prompt.txt"
DEFAULT_OUTPUT = OUTPUT_DIR / "tool_calling_dataset_synthetic.jsonl"


def load_records(path: Path) -> list[dict[str, Any]]:
    """Accept a JSON array, a single JSON record, or JSONL."""
    text = Path(path).read_text(encoding="utf-8")
    try:
        value = json.loads(text)
        rows = value if isinstance(value, list) else [value]
    except json.JSONDecodeError:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not rows or any(not isinstance(row, dict) or not row.get("messages") for row in rows):
        raise ValueError(f"{path}: expected nonempty conversation records with messages")
    return rows


def load_examples(examples: Path | str | list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Load sample conversations from a file or an in-memory list."""
    if isinstance(examples, list):
        if not examples or any(not isinstance(row, dict) or not row.get("messages") for row in examples):
            raise ValueError("Sample conversations must be nonempty records with messages")
        return examples
    return load_records(Path(examples))


def load_tools(source: Path | str | list[dict] = DEFAULT_SCHEMA) -> list[dict]:
    """Load a nonempty OpenAI function-tool list and check each parameter schema."""
    tools = json.loads(Path(source).read_text(encoding="utf-8")) if not isinstance(source, list) else source
    names: set[str] = set()
    if not isinstance(tools, list) or not tools:
        raise ValueError("Tool schema must be a nonempty OpenAI function-tool list")
    for tool in tools:
        function = tool["function"]
        if tool.get("type") != "function" or function["name"] in names:
            raise ValueError("Tool names must be unique function names")
        names.add(function["name"])
        Draft202012Validator.check_schema(function["parameters"])
    return tools


def load_seed_queries(source: Path | str | list[str] = DEFAULT_SEEDS) -> list[str]:
    queries = json.loads(Path(source).read_text(encoding="utf-8")) if not isinstance(source, list) else source
    if not isinstance(queries, list) or not queries or any(not isinstance(query, str) or not query.strip() for query in queries):
        raise ValueError("Provide a nonempty list of seed queries")
    return queries


def load_system_prompt(source: Path | str = DEFAULT_PROMPT) -> str:
    if isinstance(source, Path) or (isinstance(source, str) and Path(source).is_file()):
        prompt = Path(source).read_text(encoding="utf-8").strip()
    else:
        prompt = str(source).strip()
    if not prompt:
        raise ValueError("Provide a system prompt")
    return prompt


def prepare_inputs(
    examples: Path | str | list[dict[str, Any]],
    seed_queries: list[str],
    system_prompt: str,
    tools: list[dict],
):
    """Build one row per seed. Every row carries the full sample set."""
    import pandas as pd

    samples = load_examples(examples)
    queries = load_seed_queries(seed_queries)
    if not isinstance(system_prompt, str) or not system_prompt.strip():
        raise ValueError("Provide a system prompt")
    tool_schema = json.dumps(tools, ensure_ascii=False)
    references = json.dumps(samples, ensure_ascii=False)
    rows = [
        {
            "seed_id": index,
            "seed_query": query,
            "tool_schema": tool_schema,
            "system_prompt": system_prompt,
            "references": references,
        }
        for index, query in enumerate(queries)
    ]
    return pd.DataFrame(rows)


def validate_trajectory(value: Any, tools: list[dict], system_prompt: str) -> dict:
    """Validate structure, call arguments, and ordered call/result pairing.

    This does not establish that simulated results or answers are true.
    """
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    messages = value.get("messages")
    if not isinstance(messages, list) or len(messages) < 5 or any(not isinstance(message, dict) for message in messages):
        raise ValueError("Expected system, user, tool calls/results, and final answer")
    if messages[0] != {"role": "system", "content": system_prompt}:
        raise ValueError("System prompt must match the supplied prompt")
    if messages[1].get("role") != "user" or not isinstance(messages[1].get("content"), str) or not messages[1]["content"].strip():
        raise ValueError("Expected a nonempty user query")
    final = messages[-1]
    if final.get("role") != "assistant" or final.get("tool_calls") or not isinstance(final.get("content"), str) or not final["content"].strip():
        raise ValueError("Expected a final assistant answer")
    for message in messages:
        if isinstance(message, dict) and "tool_call_id" not in message and isinstance(message.get("tool_calls_id"), str):
            message["tool_call_id"] = message.pop("tool_calls_id")
    schemas = {tool["function"]["name"]: tool["function"]["parameters"] for tool in tools}
    seen: set[str] = set()
    pending: dict[str, str] = {}
    for message in messages[2:-1]:
        role = message.get("role")
        if role == "assistant":
            calls = message.get("tool_calls")
            if pending or not isinstance(calls, list) or not calls:
                raise ValueError("Assistant must call tools after pending results resolve")
            for call in calls:
                if not isinstance(call, dict) or call.get("type") != "function":
                    raise ValueError("Expected a function tool call")
                call_id, function = call.get("id"), call.get("function")
                if not isinstance(call_id, str) or not call_id or call_id in seen or not isinstance(function, dict):
                    raise ValueError("Tool call IDs must be nonempty and unique")
                name, arguments = function.get("name"), function.get("arguments")
                if name not in schemas or not isinstance(arguments, str):
                    raise ValueError("Unknown tool or non-string JSON arguments")
                Draft202012Validator(schemas[name]).validate(json.loads(arguments))
                seen.add(call_id)
                pending[call_id] = name
        elif role == "tool":
            call_id = message.get("tool_call_id")
            if call_id not in pending or not isinstance(message.get("content"), str):
                raise ValueError("Tool result must match a pending call and have string content")
            if message.get("name", pending[call_id]) != pending[call_id]:
                raise ValueError("Tool result name does not match its call")
            json.loads(message["content"])
            del pending[call_id]
        else:
            raise ValueError("Unexpected role in tool trajectory")
    if pending or not seen:
        raise ValueError("Unresolved or missing tool calls")
    return {
        "messages": messages,
        "source_question": messages[1]["content"],
        "iterations": sum(message.get("role") == "assistant" for message in messages),
    }


def completion_text(value: Any) -> str:
    """Return the single completion string stored by the response extractor.

    With expand_lists false, LLMResponseExtractorBlock keeps one row and stores
    the completion as a one-item list. JSONParserBlock only parses strings, so
    the runner reads that list when the parser leaves the verdict columns empty.
    """
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError("Expected exactly one model completion")
        value = value[0]
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Expected completion text")
    return value


def _missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(value != value)
    except Exception:
        return False


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if type(value).__name__ == "bool_":
        return bool(value)
    raise ValueError("Review must include boolean consistent and nonempty reason")


def _as_score(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("Review score must be a finite number in [0, 1]")
    try:
        score = float(value)
    except (TypeError, ValueError):
        raise ValueError("Review score must be a finite number in [0, 1]") from None
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("Review score must be a finite number in [0, 1]")
    return score


def parse_review(raw: str) -> dict:
    """Fail closed on missing, malformed, or incorrectly typed reviewer verdicts."""
    verdict = json.loads(raw)
    if not isinstance(verdict, dict):
        raise ValueError("Review must be an object")
    return _verdict(verdict.get("score"), verdict.get("consistent"), verdict.get("reason"))


def _verdict(score: Any, consistent: Any, reason: Any) -> dict:
    parsed_score = _as_score(score)
    parsed_consistent = _as_bool(consistent)
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("Review must include boolean consistent and nonempty reason")
    return {
        "quality_score": parsed_score,
        "quality_consistent": parsed_consistent,
        "quality_reason": reason,
    }


def verdict_from_row(row: dict) -> dict:
    """Use JSONParserBlock columns when they are present, otherwise the raw completion."""
    score, consistent, reason = row.get("score"), row.get("consistent"), row.get("reason")
    if not any(_missing(value) for value in (score, consistent, reason)):
        return _verdict(score, consistent, reason)
    return parse_review(completion_text(row.get("extract_review_content")))


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")
    temporary.replace(path)


def run(
    examples: Path | str | list[dict[str, Any]],
    seed_queries: list[str],
    system_prompt: str,
    output: Path,
    *,
    model: str,
    api_key: str | None = None,
    api_base: str | None = None,
    tools: list[dict] | None = None,
    schema_path: Path | str = DEFAULT_SCHEMA,
    variants_per_seed: int = 2,
    max_tokens: int = 6000,
    quality_check: bool = True,
    min_quality_score: float = 0.8,
) -> dict:
    """Run generation, structural checks, optional review, and the quality filter.

    Only the generation and review LLMChatBlocks contact the teacher. Tool
    results in accepted rows are simulated.
    """
    import pandas as pd
    from sdg_hub import Flow

    if not model:
        raise ValueError("Configure the teacher model before running offline-mcp-synthesis")
    if isinstance(variants_per_seed, bool) or not isinstance(variants_per_seed, int) or variants_per_seed < 1:
        raise ValueError("variants_per_seed must be a positive integer")
    if isinstance(min_quality_score, bool) or not isinstance(min_quality_score, (int, float)) or not 0 <= float(min_quality_score) <= 1:
        raise ValueError("min_quality_score must be in [0, 1]")
    output = Path(output)
    rejected_path = output.with_suffix(".rejected.jsonl")
    generated_path = output.parent / "tool_calling_dataset_generated.jsonl"
    inputs_on_disk = {Path(schema_path).resolve()}
    if not isinstance(examples, list):
        inputs_on_disk.add(Path(examples).resolve())
    if any(path.resolve() in inputs_on_disk for path in (output, rejected_path, generated_path)):
        raise ValueError("Output must differ from the sample conversations and the tool schema")

    loaded_tools = load_tools(schema_path) if tools is None else load_tools(tools)
    frame = prepare_inputs(examples, seed_queries, system_prompt, loaded_tools)
    generation = Flow.from_yaml(str(GENERATION_FLOW))
    dataset_errors = generation.validate_dataset(frame)
    if dataset_errors:
        raise ValueError("; ".join(dataset_errors))
    generation.set_model_config(model=model, api_key=api_key, api_base=api_base, max_tokens=max_tokens)
    result = generation.generate(frame, runtime_params={"expand_seeds": {"num_samples": variants_per_seed}})
    expected = len(frame) * variants_per_seed
    if len(result) != expected:
        raise RuntimeError("Teacher/extractor changed row count; no output written")
    result["variant"] = result.groupby("seed_id", sort=False).cumcount() + 1

    rows = result.to_dict("records")
    candidates: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    questions: set[str] = set()
    for candidate_id, row in enumerate(rows):
        raw = row.get("extract_conversation_content")
        try:
            record = validate_trajectory(json.loads(completion_text(raw)), loaded_tools, system_prompt)
            question_key = record["source_question"].strip().casefold()
            if question_key in questions:
                raise ValueError("Duplicate generated query")
            questions.add(question_key)
            record.update(
                mode="offline_mcp_synthesis",
                teacher_model=model,
                synthetic_tool_results=True,
                tools=loaded_tools,
                seed_query=row["seed_query"],
                variant=int(row["variant"]),
                quality_checked=False,
            )
            candidates.append({
                "candidate_id": candidate_id,
                "record": record,
                "candidate_json": json.dumps({"messages": record["messages"]}, ensure_ascii=False),
                "tool_schema": row["tool_schema"],
            })
        except (ValueError, TypeError, KeyError, ValidationError, json.JSONDecodeError) as error:
            rejected.append({
                "stage": "structure",
                "seed_query": row.get("seed_query"),
                "variant": row.get("variant"),
                "error": str(error),
                "raw_response": raw,
            })

    accepted = [candidate["record"] for candidate in candidates]
    if quality_check and candidates:
        review = Flow.from_yaml(str(REVIEW_FLOW))
        review_frame = pd.DataFrame(candidates)
        review_errors = review.validate_dataset(review_frame)
        if review_errors:
            raise ValueError("; ".join(review_errors))
        review.set_model_config(model=model, api_key=api_key, api_base=api_base)
        reviews = review.generate(review_frame)
        if len(reviews) != len(candidates) or set(reviews["candidate_id"]) != {candidate["candidate_id"] for candidate in candidates}:
            raise RuntimeError("Reviewer changed candidate count or IDs; no output written")
        graded: list[dict[str, Any]] = []
        for row in reviews.to_dict("records"):
            record = dict(row["record"])
            try:
                verdict = verdict_from_row(row)
                record.update(verdict, quality_checked=True, quality_reviewer_model=model)
                graded.append({"candidate_id": row["candidate_id"], "record": record, **verdict})
            except (ValueError, TypeError, json.JSONDecodeError) as error:
                rejected.append({
                    "stage": "review",
                    "error": str(error),
                    "record": record,
                    "raw_review": row.get("extract_review_content"),
                })
        accepted = []
        if graded:
            from sdg_hub.core.utils.error_handling import EmptyDatasetError

            filter_flow = Flow.from_yaml(str(FILTER_FLOW))
            graded_frame = pd.DataFrame(graded)
            filter_errors = filter_flow.validate_dataset(graded_frame)
            if filter_errors:
                raise ValueError("; ".join(filter_errors))
            try:
                # Runtime override replaces filter_value after the block has already
                # wrapped the YAML value in a list, so pass a one-item list.
                filtered = filter_flow.generate(graded_frame, runtime_params={
                    "keep_high_quality": {"filter_value": [float(min_quality_score)]},
                })
                accepted_ids = set(filtered["candidate_id"])
            except EmptyDatasetError:
                accepted_ids = set()
            for row in graded:
                if row["candidate_id"] in accepted_ids:
                    accepted.append(row["record"])
                else:
                    rejected.append({"stage": "quality", "error": row["quality_reason"], "record": row["record"]})

    _write_jsonl(output, accepted)
    _write_jsonl(rejected_path, rejected)
    _write_jsonl(generated_path, [*accepted, *rejected])
    return {
        "requested": expected,
        "accepted": len(accepted),
        "rejected": len(rejected),
        "quality_check": quality_check,
        "min_quality_score": float(min_quality_score),
        "generated": len(accepted) + len(rejected),
        "output": str(output),
        "generated_output": str(generated_path),
        "rejected_output": str(rejected_path),
    }


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no"}


def main() -> int:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    model = os.getenv("TEACHER_MODEL_NAME") or os.getenv("MODEL", "")
    if not model:
        print("Skipping offline-mcp-synthesis: set TEACHER_MODEL_NAME in .env")
        return 0
    report = run(
        Path(os.getenv("EXAMPLES_PATH", str(DEFAULT_EXAMPLES))),
        load_seed_queries(Path(os.getenv("SEEDS_PATH", str(DEFAULT_SEEDS)))),
        load_system_prompt(Path(os.getenv("SYSTEM_PROMPT_PATH", str(DEFAULT_PROMPT)))),
        Path(os.getenv("OUTPUT_PATH", str(DEFAULT_OUTPUT))),
        model=model,
        api_base=os.getenv("TEACHER_MODEL_URL") or os.getenv("API_BASE"),
        api_key=os.getenv("TEACHER_API_KEY") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY"),
        schema_path=Path(os.getenv("TOOL_SCHEMA_PATH", str(DEFAULT_SCHEMA))),
        variants_per_seed=int(os.getenv("VARIANTS_PER_SEED", "2")),
        quality_check=_env_flag("QUALITY_CHECK", True),
        min_quality_score=float(os.getenv("MIN_QUALITY_SCORE", "0.8")),
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
