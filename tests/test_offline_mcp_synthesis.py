"""Exercise the offline-mcp-synthesis flows with the teacher completion mocked."""
import copy
import json
import socket
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from jsonschema import ValidationError

from offline_mcp_synthesis import (
    ROOT,
    completion_text,
    load_examples,
    load_seed_queries,
    load_system_prompt,
    load_tools,
    parse_review,
    prepare_inputs,
    run,
    validate_trajectory,
)

SYSTEM_PROMPT = load_system_prompt(ROOT / "input" / "system_prompt.txt")
SEEDS = ["Inspect ORD-1001.", "Why is ORD-1003 waiting?"]


def trajectory(question="Please inspect ORD-1001 for me."):
    return {"messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_1", "type": "function", "function": {
                "name": "get_order", "arguments": '{"order_id":"ORD-1001"}'}}
        ]},
        {"role": "tool", "tool_call_id": "call_1", "name": "get_order",
         "content": '{"id":"ORD-1001","status":"paid"}'},
        {"role": "assistant", "content": "ORD-1001 is paid."},
    ]}


def completion_batch(payloads):
    """Match LLMChatBlock._generate_sync: one list of message dicts per row."""
    return [[{"role": "assistant", "content": payload if isinstance(payload, str) else json.dumps(payload)}]
            for payload in payloads]


class OfflineMcpSynthesisTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.tools = load_tools()
        self.examples = load_examples(ROOT / "input" / "examples.jsonl")

    def test_pack_files_and_flow_metadata(self):
        from sdg_hub import Flow

        self.assertEqual(len(self.examples), 3)
        self.assertEqual(len(load_seed_queries()), 5)
        self.assertIn("paid orders", SYSTEM_PROMPT)
        flows = {
            "offline-mcp-synthesis": ROOT / "generation" / "flow.yaml",
            "offline-mcp-synthesis-review": ROOT / "review" / "flow.yaml",
            "offline-mcp-synthesis-filter": ROOT / "filter" / "flow.yaml",
        }
        for flow_id, path in flows.items():
            flow = Flow.from_yaml(str(path))
            self.assertEqual(flow.metadata.id, flow_id)
            self.assertTrue(flow.metadata.author)
            self.assertGreaterEqual(len(flow.metadata.tags), 1)
            self.assertIn("offline", flow.metadata.tags)
        generation = Flow.from_yaml(str(flows["offline-mcp-synthesis"]))
        self.assertEqual(
            [type(block).__name__ for block in generation.blocks],
            ["RowMultiplierBlock", "PromptBuilderBlock", "LLMChatBlock", "LLMResponseExtractorBlock"],
        )
        self.assertFalse(generation.blocks[-1].expand_lists)
        review = Flow.from_yaml(str(flows["offline-mcp-synthesis-review"]))
        self.assertEqual(review.blocks[-1].__class__.__name__, "JSONParserBlock")
        self.assertEqual(list(review.blocks[-1].output_cols), ["score", "consistent", "reason"])
        frame = prepare_inputs(self.examples, SEEDS, SYSTEM_PROMPT, self.tools)
        self.assertEqual(generation.validate_dataset(frame), [])
        self.assertTrue(generation.validate_dataset(frame.drop(columns=["references"])))

    def test_prepare_inputs_keeps_every_sample_on_each_seed(self):
        frame = prepare_inputs(self.examples, SEEDS, SYSTEM_PROMPT, self.tools)
        self.assertEqual(frame["seed_query"].tolist(), SEEDS)
        references = json.loads(frame.iloc[0]["references"])
        self.assertEqual(references, json.loads(frame.iloc[1]["references"]))
        self.assertEqual(len(references), 3)
        self.assertNotIn("source_files", frame.columns)
        with self.assertRaises(ValueError):
            prepare_inputs(self.examples, ["  "], SYSTEM_PROMPT, self.tools)
        with self.assertRaises(ValueError):
            prepare_inputs([{"messages": []}], SEEDS, SYSTEM_PROMPT, self.tools)

    def test_json_parser_lifts_review_fields(self):
        import pandas as pd
        from sdg_hub import Flow

        review = Flow.from_yaml(str(ROOT / "review" / "flow.yaml"))
        parser = review.blocks[-1]
        parsed = parser(pd.DataFrame({
            "extract_review_content": [json.dumps({
                "score": 0.91, "consistent": True, "reason": "Stock covers the order."
            })]
        }))
        self.assertEqual(parsed.iloc[0]["score"], 0.91)
        self.assertEqual(bool(parsed.iloc[0]["consistent"]), True)
        self.assertIn("Stock", parsed.iloc[0]["reason"])

    def test_unknown_tools_and_bad_arguments(self):
        for function in (
            {"name": "unknown", "arguments": "{}"},
            {"name": "get_order", "arguments": "{}"},
            {"name": "get_order", "arguments": '{"order_id":42}'},
            {"name": "get_order", "arguments": '{"order_id":"x","extra":true}'},
            {"name": "get_order", "arguments": {"order_id": "x"}},
        ):
            row = trajectory()
            row["messages"][2]["tool_calls"][0]["function"] = function
            with self.assertRaises((ValueError, ValidationError)):
                validate_trajectory(row, self.tools, SYSTEM_PROMPT)

    def test_call_result_integrity(self):
        base = trajectory()
        cases = []
        row = copy.deepcopy(base)
        row["messages"][3]["tool_call_id"] = "orphan"
        cases.append(row)
        row = copy.deepcopy(base)
        del row["messages"][3]
        cases.append(row)
        row = copy.deepcopy(base)
        row["messages"].insert(4, copy.deepcopy(row["messages"][3]))
        cases.append(row)
        row = copy.deepcopy(base)
        row["messages"][2]["tool_calls"] *= 2
        cases.append(row)
        row = copy.deepcopy(base)
        row["messages"][3]["name"] = "create_shipment"
        cases.append(row)
        row = copy.deepcopy(base)
        row["messages"][0]["content"] = "different"
        cases.append(row)
        for row in cases:
            with self.assertRaises(ValueError):
                validate_trajectory(row, self.tools, SYSTEM_PROMPT)
        self.assertEqual(validate_trajectory(base, self.tools, SYSTEM_PROMPT)["iterations"], 2)
        aliased = copy.deepcopy(base)
        aliased["messages"][3]["tool_calls_id"] = aliased["messages"][3].pop("tool_call_id")
        normalized = validate_trajectory(aliased, self.tools, SYSTEM_PROMPT)
        self.assertEqual(normalized["messages"][3]["tool_call_id"], "call_1")
        self.assertNotIn("tool_calls_id", normalized["messages"][3])
        self.assertEqual(completion_text([json.dumps(base)]), json.dumps(base))

    def test_review_verdicts_fail_closed(self):
        for verdict in (
            {},
            {"score": True, "consistent": True, "reason": "ok"},
            {"score": 0.9, "consistent": "true", "reason": "ok"},
            {"score": 1.2, "consistent": True, "reason": "ok"},
            {"score": float("nan"), "consistent": True, "reason": "ok"},
            {"score": 0.4, "consistent": True, "reason": "  "},
        ):
            with self.assertRaises(ValueError):
                parse_review(json.dumps(verdict))

    def test_pipeline_writes_one_variant_without_network(self):
        from sdg_hub.core.blocks import LLMChatBlock

        output = self.root / "synthetic.jsonl"
        examples_before = (ROOT / "input" / "examples.jsonl").read_bytes()
        with patch.object(LLMChatBlock, "_generate_sync", return_value=completion_batch([
            trajectory("Where is ORD-1001?"),
            "not JSON",
        ])) as completion, patch.object(socket.socket, "connect", side_effect=AssertionError("network")):
            report = run(
                self.examples, SEEDS[:1], SYSTEM_PROMPT, output,
                model="openai/test", api_key="test", tools=self.tools,
                variants_per_seed=2, quality_check=False,
            )
        self.assertEqual((report["requested"], report["accepted"], report["rejected"]), (2, 1, 1))
        prompts, kwargs = completion.call_args.args
        self.assertNotIn("tools", kwargs)
        self.assertNotIn("functions", kwargs)
        self.assertIn("Sample conversations", prompts[0][1]["content"])
        self.assertIn("ORD-1003", prompts[0][1]["content"])
        record = load_examples(output)[0]
        self.assertEqual(record["mode"], "offline_mcp_synthesis")
        self.assertTrue(record["synthetic_tool_results"])
        self.assertFalse(record["quality_checked"])
        self.assertEqual(record["variant"], 1)
        self.assertEqual(record["tools"], self.tools)
        self.assertEqual((ROOT / "input" / "examples.jsonl").read_bytes(), examples_before)
        self.assertEqual(len(Path(report["rejected_output"]).read_text().splitlines()), 1)
        generated = [json.loads(line) for line in Path(report["generated_output"]).read_text().splitlines() if line.strip()]
        self.assertEqual(report["generated"], 2)
        self.assertEqual(len(generated), 2)
        self.assertFalse(generated[0]["quality_checked"])
        self.assertIn("messages", generated[0])
        self.assertNotIn("messages", generated[1])

    def test_duplicate_questions_and_all_structural_rejects(self):
        from sdg_hub.core.blocks import LLMChatBlock

        output = self.root / "output.jsonl"
        valid = trajectory("Same question")
        with patch.object(LLMChatBlock, "_generate_sync", return_value=completion_batch([valid, valid])):
            report = run(
                self.examples, SEEDS[:1], SYSTEM_PROMPT, output,
                model="openai/test", tools=self.tools, quality_check=False,
            )
        self.assertEqual((report["accepted"], report["rejected"]), (1, 1))
        self.assertEqual(report["generated"], 2)
        generated = [json.loads(line) for line in Path(report["generated_output"]).read_text().splitlines() if line.strip()]
        self.assertEqual(generated[0]["messages"][1]["content"], "Same question")
        self.assertEqual(generated[1]["stage"], "structure")
        self.assertEqual(generated[1]["error"], "Duplicate generated query")
        with patch.object(LLMChatBlock, "_generate_sync", return_value=completion_batch(["bad"])):
            report = run(
                self.examples, SEEDS[:1], SYSTEM_PROMPT, output,
                model="openai/test", tools=self.tools, variants_per_seed=1, quality_check=False,
            )
        self.assertEqual((report["accepted"], report["rejected"], report["generated"]), (0, 1, 1))
        self.assertEqual(len(Path(report["generated_output"]).read_text().splitlines()), 1)
        self.assertTrue(output.with_suffix(".rejected.jsonl").exists())

    def test_quality_filter_drops_the_shipment_contradiction(self):
        from sdg_hub.core.blocks import LLMChatBlock

        bad = trajectory("Ship ORD-1002 via FedEx.")
        bad["messages"][2]["tool_calls"][0]["function"] = {
            "name": "create_shipment",
            "arguments": '{"order_id":"ORD-1002","carrier":"FedEx"}',
        }
        bad["messages"][3]["name"] = "create_shipment"
        bad["messages"][3]["content"] = '{"error":"Insufficient inventory for SKU-MON-27"}'
        bad["messages"][-1]["content"] = "FedEx shipment created for ORD-1002."
        output = self.root / "reviewed.jsonl"
        observed = []

        def complete(prompts, kwargs):
            self.assertNotIn("tools", kwargs)
            observed.append(prompts)
            if len(observed) == 1:
                return completion_batch([trajectory("Look up ORD-1001."), bad])
            self.assertIn("Insufficient inventory", prompts[1][1]["content"])
            return completion_batch([
                {"score": 0.95, "consistent": True, "reason": "Paid state matches the answer."},
                {"score": 0, "consistent": False, "reason": "ORD-1002 shipment failed, answer claims success."},
            ])

        with patch.object(LLMChatBlock, "_generate_sync", side_effect=complete), patch.object(
            socket.socket, "connect", side_effect=AssertionError("network")
        ):
            report = run(self.examples, SEEDS[:1], SYSTEM_PROMPT, output, model="openai/test", tools=self.tools)
        self.assertEqual(len(observed), 2)
        self.assertEqual((report["accepted"], report["rejected"]), (1, 1))
        accepted = load_examples(output)[0]
        self.assertTrue(accepted["quality_checked"])
        self.assertEqual(accepted["quality_score"], 0.95)
        self.assertEqual(accepted["quality_reviewer_model"], "openai/test")
        rejected = json.loads(Path(report["rejected_output"]).read_text())
        self.assertEqual(rejected["stage"], "quality")
        self.assertIn("shipment failed", rejected["error"])
        generated = [json.loads(line) for line in Path(report["generated_output"]).read_text().splitlines() if line.strip()]
        self.assertEqual(generated, [accepted, rejected])
        self.assertIn("FedEx shipment created", generated[1]["record"]["messages"][-1]["content"])

    def test_threshold_and_invalid_review_can_reject_everything(self):
        from sdg_hub.core.blocks import LLMChatBlock

        generation = completion_batch([trajectory("one"), trajectory("two")])
        reviews = completion_batch([
            {"score": 0.5, "consistent": True, "reason": "Weak response"},
            "not JSON",
        ])
        with patch.object(LLMChatBlock, "_generate_sync", side_effect=[generation, reviews]):
            report = run(
                self.examples, SEEDS[:1], SYSTEM_PROMPT, self.root / "all_rejected.jsonl",
                model="openai/test", tools=self.tools,
            )
        self.assertEqual((report["accepted"], report["rejected"]), (0, 2))
        generated = [json.loads(line) for line in Path(report["generated_output"]).read_text().splitlines() if line.strip()]
        self.assertEqual(len(generated), 2)
        self.assertTrue(all(row["stage"] in {"review", "quality"} for row in generated))

        generation = completion_batch([trajectory("alpha"), trajectory("beta")])
        reviews = completion_batch([
            {"score": 1, "consistent": False, "reason": "Contradiction despite high score"},
            {"score": 0.95, "consistent": True, "reason": "Coherent but below the custom threshold"},
        ])
        with patch.object(LLMChatBlock, "_generate_sync", side_effect=[generation, reviews]):
            report = run(
                self.examples, SEEDS[:1], SYSTEM_PROMPT, self.root / "threshold.jsonl",
                model="openai/test", tools=self.tools, min_quality_score=0.99,
            )
        self.assertEqual((report["accepted"], report["rejected"]), (0, 2))
        generated = [json.loads(line) for line in Path(report["generated_output"]).read_text().splitlines() if line.strip()]
        self.assertEqual(len(generated), 2)
        self.assertTrue(all(row["stage"] == "quality" for row in generated))
        self.assertEqual(report["min_quality_score"], 0.99)

    def test_cannot_overwrite_inputs(self):
        with self.assertRaisesRegex(ValueError, "Output must differ"):
            run(ROOT / "input" / "examples.jsonl", SEEDS, SYSTEM_PROMPT, ROOT / "input" / "examples.jsonl", model="openai/test")
        with self.assertRaisesRegex(ValueError, "variants_per_seed"):
            run(self.examples, SEEDS, SYSTEM_PROMPT, self.root / "out.jsonl", model="openai/test", variants_per_seed=0)


if __name__ == "__main__":
    unittest.main()
