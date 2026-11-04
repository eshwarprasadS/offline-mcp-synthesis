# offline-mcp-synthesis

Teacher-only synthesis of tool-calling training conversations. A language model writes new questions and complete simulated transcripts from four caller-supplied inputs: sample conversations, seed queries, a system prompt, and a tool schema.

The flow does not start an MCP server, register tools with the model, execute functions, or call an agent. The only network call is to the teacher endpoint. Tool results in the generated conversations are written by the model.

## Goal

Produce `n` synthetic conversations that look like real tool-use traces:

- the system prompt is the one the caller supplied
- the user question is a new variant of a seed
- assistant turns call only tools declared in the schema
- tool results are plausible JSON for that new scenario
- the final answer agrees with those simulated results

`n` is `len(seed_queries) * variants_per_seed`. Quality checks can accept fewer rows. Zero accepted rows is a valid result.

## Inputs

Nothing is inferred from another dataset's `source_question` or from a pair of source files. Each run receives:

| Input | Required | Form |
| --- | --- | --- |
| Sample conversations | Yes | JSONL, a JSON array, a single JSON record, or an in-memory list |
| Seed queries | Yes | Nonempty list of strings |
| System prompt | Yes | One string for the whole run |
| Tool schema | Yes | OpenAI function-tool list, from a JSON file or in memory |

### Sample conversations

Each sample is one record with a `messages` array. A usable sample contains a system message, a user message, at least one assistant message with `tool_calls`, matching `tool` results, and a final assistant answer.

Every sample is included, in full, as prompt context for every seed. Samples are data. The teacher must not treat their text as new instructions. Samples can disagree with each other, and a sample's final answer can disagree with its own tool result. The teacher must not copy that contradiction.

The prompt calls these "sample conversations." It does not name an upstream pipeline.

### Seed queries

The caller supplies the list. A blank string is rejected. Each seed becomes one dataframe row before expansion. `variants_per_seed` (default 2) copies that row so the teacher writes that many distinct conversations for the same intent.

### System prompt

One prompt per run. The teacher must copy it verbatim into `messages[0]`. Structural validation rejects a transcript whose first message is anything else.

### Tool schema

A nonempty list of unique OpenAI function tools. `parameters` must be a valid JSON Schema. The schema is inserted into the prompt as a contract. No tool is registered with the teacher, and no tool is executed.

## Server pack

MCP-related inputs travel together as one pack: server details, tool schema, system prompt, seed queries, and sample conversations. The flow machinery does not change when another MCP server is plugged in. The pack does. Domain sentences in the generation and review prompts (payment state, stock, shipments) belong to the pack and are rewritten with it.

For each MCP server, rewrite these two prompt files:

- `generation/prompts/synthesize.yaml`
- `review/prompts/review.yaml`

The flow still does not connect to the server. The details below describe the server that this pack's schema and samples represent. They are the current Order Ops sample, not a fixed part of the flow.

### MCP server details

| Field | Order Ops sample |
| --- | --- |
| Name | Order Ops |
| Instructions | Order and inventory tools for an integration runtime. Use these to look up orders, check stock, and create shipments. |
| Tools | `list_orders`, `get_order`, `check_inventory`, `create_shipment` |

Tool behavior this pack expects the teacher to simulate:

- `list_orders` returns orders, optionally filtered by `status` (`paid`, `pending`, or `shipped`).
- `get_order` returns one order, or an error when the id is unknown.
- `check_inventory` returns on-hand quantity for a SKU, or an error when the SKU is unknown.
- `create_shipment` ships only a `paid` order with enough stock. Success decrements `on_hand`, sets the order to `shipped`, and returns a shipment id. An unpaid order or a short stock position returns an error and changes nothing.

Catalog the samples refer to:

| Order | Customer | SKU | Qty | Status | Ship to |
| --- | --- | --- | --- | --- | --- |
| ORD-1001 | Acme Retail | SKU-LAP-14 | 2 | paid | Austin, TX |
| ORD-1002 | Northwind | SKU-MON-27 | 5 | paid | Seattle, WA |
| ORD-1003 | Contoso | SKU-KB-MECH | 10 | pending | Boston, MA |

| SKU | Name | On hand | Warehouse |
| --- | --- | --- | --- |
| SKU-LAP-14 | 14in Ultrabook | 12 | DFW |
| SKU-MON-27 | 27in Monitor | 3 | SEA |
| SKU-KB-MECH | Mechanical Keyboard | 40 | BOS |

### System prompt

```text
You are an order-operations assistant. Use the supplied tools to inspect orders and inventory before making decisions. Only paid orders with sufficient stock can ship. Base your final answer on tool results, including errors.
```

### Seed queries

```json
[
  "List all paid orders and tell me which ones we can fulfill from current stock.",
  "Look up order ORD-1001, check inventory for its SKU, and create a UPS shipment if possible.",
  "Can we ship ORD-1002 today? Check inventory and create a FedEx shipment if stock allows.",
  "Why can't we ship ORD-1003 yet? Inspect the order and inventory.",
  "Summarize on-hand inventory for every SKU referenced by paid orders."
]
```

### Sample messages

Two consistent transcripts. Tool results agree with the final answer.

Shipment allowed. ORD-1001 is paid and SKU-LAP-14 has stock for quantity 2:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are an order-operations assistant. Use the supplied tools to inspect orders and inventory before making decisions. Only paid orders with sufficient stock can ship. Base your final answer on tool results, including errors."
    },
    {
      "role": "user",
      "content": "Look up order ORD-1001, check inventory for its SKU, and create a UPS shipment if possible."
    },
    {
      "role": "assistant",
      "content": null,
      "tool_calls": [
        {
          "id": "call_1",
          "type": "function",
          "function": {
            "name": "get_order",
            "arguments": "{\"order_id\": \"ORD-1001\"}"
          }
        }
      ]
    },
    {
      "role": "tool",
      "tool_call_id": "call_1",
      "name": "get_order",
      "content": "{\"id\":\"ORD-1001\",\"customer\":\"Acme Retail\",\"sku\":\"SKU-LAP-14\",\"qty\":2,\"status\":\"paid\",\"ship_to\":\"Austin, TX\"}"
    },
    {
      "role": "assistant",
      "content": null,
      "tool_calls": [
        {
          "id": "call_2",
          "type": "function",
          "function": {
            "name": "check_inventory",
            "arguments": "{\"sku\": \"SKU-LAP-14\"}"
          }
        }
      ]
    },
    {
      "role": "tool",
      "tool_call_id": "call_2",
      "name": "check_inventory",
      "content": "{\"sku\":\"SKU-LAP-14\",\"name\":\"14in Ultrabook\",\"on_hand\":12,\"warehouse\":\"DFW\"}"
    },
    {
      "role": "assistant",
      "content": null,
      "tool_calls": [
        {
          "id": "call_3",
          "type": "function",
          "function": {
            "name": "create_shipment",
            "arguments": "{\"order_id\": \"ORD-1001\", \"carrier\": \"UPS\"}"
          }
        }
      ]
    },
    {
      "role": "tool",
      "tool_call_id": "call_3",
      "name": "create_shipment",
      "content": "{\"shipment_id\":\"SHP-0001\",\"order_id\":\"ORD-1001\",\"carrier\":\"UPS\",\"sku\":\"SKU-LAP-14\",\"qty\":2,\"warehouse\":\"DFW\"}"
    },
    {
      "role": "assistant",
      "content": "ORD-1001 shipped via UPS after confirming inventory for SKU-LAP-14."
    }
  ]
}
```

Shipment refused. ORD-1003 is pending, so stock does not authorize a shipment:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are an order-operations assistant. Use the supplied tools to inspect orders and inventory before making decisions. Only paid orders with sufficient stock can ship. Base your final answer on tool results, including errors."
    },
    {
      "role": "user",
      "content": "Why can't we ship ORD-1003 yet? Inspect the order and inventory."
    },
    {
      "role": "assistant",
      "content": null,
      "tool_calls": [
        {
          "id": "call_1",
          "type": "function",
          "function": {
            "name": "get_order",
            "arguments": "{\"order_id\": \"ORD-1003\"}"
          }
        }
      ]
    },
    {
      "role": "tool",
      "tool_call_id": "call_1",
      "name": "get_order",
      "content": "{\"id\":\"ORD-1003\",\"customer\":\"Contoso\",\"sku\":\"SKU-KB-MECH\",\"qty\":10,\"status\":\"pending\",\"ship_to\":\"Boston, MA\"}"
    },
    {
      "role": "assistant",
      "content": null,
      "tool_calls": [
        {
          "id": "call_2",
          "type": "function",
          "function": {
            "name": "check_inventory",
            "arguments": "{\"sku\": \"SKU-KB-MECH\"}"
          }
        }
      ]
    },
    {
      "role": "tool",
      "tool_call_id": "call_2",
      "name": "check_inventory",
      "content": "{\"sku\":\"SKU-KB-MECH\",\"name\":\"Mechanical Keyboard\",\"on_hand\":40,\"warehouse\":\"BOS\"}"
    },
    {
      "role": "assistant",
      "content": "ORD-1003 is still pending payment. Inventory for SKU-KB-MECH is available, but only paid orders can be shipped."
    }
  ]
}
```

One sample is intentionally inconsistent and is called out in the generation prompt as a pattern to avoid. `create_shipment` for ORD-1002 returns `{"error":"Insufficient inventory for SKU-MON-27"}` because quantity 5 exceeds on-hand 3, but the final assistant message says the FedEx shipment was created. A new transcript must not copy that contradiction.

### What changes with another MCP server

Replace the pack together:

- server name, instructions, tool behavior, and any catalog the samples rely on
- `input/tool_schema.json`
- `input/system_prompt.txt`
- `input/seeds.json`
- `input/examples.jsonl`
- `generation/prompts/synthesize.yaml`
- `review/prompts/review.yaml`

`generation/prompts/synthesize.yaml` and `review/prompts/review.yaml` hold the domain rules (for Order Ops: payment state, stock, shipments, and the ORD-1002 negative example). The `flow.yaml` block lists do not mention a server. Argument validation follows whatever JSON Schema is in the new pack. Structural checks, the score threshold, and the output files stay as they are.

## Row sent to generation

`prepare_inputs` builds one row per seed:

| Column | Contents |
| --- | --- |
| `seed_id` | Zero-based index of the seed |
| `seed_query` | The seed string |
| `tool_schema` | JSON string of the function-tool list |
| `system_prompt` | The caller-supplied prompt |
| `references` | JSON string of the sample conversations |

There is no `source_files` column and no pairing of two input files.

## SDG Hub layout

These are local custom flows, loaded with `Flow.from_yaml`. They are not built-in flows under `src/sdg_hub/flows/`. Each flow is one directory with `flow.yaml` and, when it has prompts, a `prompts/` folder. `prompt_config_path` is resolved from the directory that contains `flow.yaml`.

```text
offline-mcp-synthesis/
  generation/flow.yaml
  generation/prompts/synthesize.yaml
  review/flow.yaml
  review/prompts/review.yaml
  filter/flow.yaml
```

Python runs the three flows in order because structural checks sit between generation and review, and rejected rows must be saved. `ColumnValueFilterBlock` only drops rows, so the runner compares candidate ids before and after the filter flow and writes the dropped ones itself.

Each `flow.yaml` has `metadata` and `blocks`. Required metadata: `name`, `author`, `description`, and at least one `tag`. Also set `id`, `version`, `license`, `recommended_models`, and `dataset_requirements`. The id is lowercase words separated by hyphens. The name is a title. `set_model_config` supplies the teacher at runtime, so the YAML does not hardcode `model`, `api_key`, or `api_base`.

| Flow | id | name |
| --- | --- | --- |
| Generation | `offline-mcp-synthesis` | Offline MCP Synthesis |
| Review | `offline-mcp-synthesis-review` | Offline MCP Synthesis Review |
| Filter | `offline-mcp-synthesis-filter` | Offline MCP Synthesis Filter |

Tags for all three: `tool-calling`, `synthetic-data`, `offline`. `recommended_models.default` records one LiteLLM-style teacher, such as `openai/gpt-oss-120b`. The running model still comes from the environment through `set_model_config`.

`dataset_requirements.required_columns`:

| Flow | Required columns |
| --- | --- |
| Generation | `seed_id`, `seed_query`, `tool_schema`, `system_prompt`, `references` |
| Review | `candidate_id`, `candidate_json`, `tool_schema` |
| Filter | `candidate_id`, `record`, `quality_consistent`, `quality_score`, `quality_reason` |

`min_samples` is 1. Omit `output_columns` so the runner can read both the original columns and the block outputs. `output_columns` would keep input columns and drop intermediates, which is the wrong shape for the JSONL written afterward.

Block names are unique inside a flow. Counts and the score threshold are runtime overrides, passed to `Flow.generate` and restored when the call returns:

```python
generation.generate(inputs, runtime_params={
    "expand_seeds": {"num_samples": variants_per_seed},
})
review.set_model_config(model=model, api_key=api_key, api_base=api_base)
filter_flow.generate(graded, runtime_params={
    "keep_high_quality": {"filter_value": min_quality_score},
})
```

`num_samples` and `filter_value` are ordinary block fields, so this override is valid. `input_cols`, `output_cols`, and `block_name` stay in the YAML.

Call `set_model_config` on each flow that contains an `LLMChatBlock`. Pass `max_tokens=6000` only to the generation flow. The review flow keeps the `max_tokens: 1500` written on its `LLMChatBlock`. One `set_model_config` call applies extra keyword arguments to every LLM block in that flow.

## Generation flow

```text
expand_seeds            RowMultiplierBlock          num_samples default 2, shuffle false
build_synthesis_prompt  PromptBuilderBlock          prompts/synthesize.yaml
generate_conversation   LLMChatBlock                temperature 0.7, max_tokens 6000,
                                                    response_format json_object, async_mode false
extract_conversation    LLMResponseExtractorBlock   extract_content true, expand_lists false
```

`PromptBuilderBlock.input_cols` is the list `seed_query`, `tool_schema`, `system_prompt`, `references`. The template variables use those same names. The block output is a list of chat messages, which is what `LLMChatBlock` requires. The extractor writes `extract_conversation_content`. `expand_lists` stays false so one seed copy stays one row. A list of completions is how that flag adds rows, and this run asks for one transcript per copy.

`prompts/synthesize.yaml` is a list of `{role, content}` messages rendered with Jinja. For the Order Ops pack it tells the teacher to:

- write one new user query from the seed, varying phrasing or scenario
- use only the supplied function names and parameters, with `arguments` as a JSON string
- simulate tool results that stay consistent inside the new scenario
- alternate tool-call turns and matching tool results, then a final answer supported by those results
- refuse a shipment when the simulated order is unpaid or stock is short
- never claim success after a simulated error

The Order Ops negative example lives in that prompt file: `create_shipment` for ORD-1002 returns insufficient inventory, and the sample answer still says a FedEx shipment was created. Another pack replaces the prompt file.

Only `generate_conversation` contacts the model. If the extractor changes the row count, the run stops and writes nothing. An empty frame raises `EmptyDatasetError`.

## Structural validation

No registered block checks tool-call pairing or JSON Schema. That check stays in Python, between the generation flow and the review flow. It does not establish that simulated facts are true. Adding a new block for it is out of scope. `JSONParserBlock` only lifts JSON fields into columns, so it is not used on the transcript.

A transcript is accepted only when:

- it is one JSON object with a `messages` list of at least five messages
- message 0 is exactly `{role: system, content: <the supplied system prompt>}`
- message 1 is a nonempty user string
- the last message is an assistant answer with no tool calls
- middle messages are assistant tool-call turns or tool results
- an assistant turn contains at least one function call and occurs only after earlier calls have results
- each call has a unique nonempty id, `type: function`, a known tool name, and arguments that validate against that tool's JSON Schema
- each tool message matches one pending call id, uses the same tool name when a name is present, and has JSON content
- no call is left unresolved, and at least one call exists

The user text becomes `source_question`. Two candidates with the same question, compared case-insensitively after trimming, are duplicates. The later one is rejected.

Failures at this stage are recorded with `stage: structure` and the raw model text.

## Review flow

Quality review is on by default. It is a second teacher request, one call per structurally valid candidate. Temperature is 0. It does not execute tools.

```text
build_review_prompt   PromptBuilderBlock          prompts/review.yaml
review_conversation   LLMChatBlock                temperature 0, max_tokens 1500,
                                                  response_format json_object, async_mode false
extract_review        LLMResponseExtractorBlock   extract_content true, expand_lists false
parse_review          JSONParserBlock             fields score, consistent, reason
```

`build_review_prompt` reads `candidate_json` and `tool_schema`. `JSONParserBlock` reads `extract_review_content` and writes columns `score`, `consistent`, and `reason`. That is the standard step after an extractor when the model returns one JSON object.

The Order Ops `prompts/review.yaml` asks the teacher to check:

- identifiers, quantities, stock, payment state, and errors agree across the user request, arguments, simulated results, and the final answer
- later results respect earlier mutations, such as a successful shipment reducing stock and marking the order shipped
- a claimed shipment is backed by a successful `create_shipment` result for that order
- the answer acknowledges failures
- the conversation is useful and responds to the query

The reviewer returns exactly:

```json
{"score": 0.0, "consistent": false, "reason": "specific evidence"}
```

Python then fail-closes. `score` must be a finite number in `[0, 1]`. `consistent` must be a boolean. `reason` must be a nonempty string. A parser failure or a bad type is `stage: review`. Rows that pass are copied to `quality_score`, `quality_consistent`, and `quality_reason` for the filter flow.

A factual contradiction requires `consistent: false` and `score` at most 0.2. A coherent failure, such as an answer that correctly explains insufficient stock, can score highly. Invented identifiers are allowed when every use inside that one transcript agrees. A high score does not verify any external system.

`set_model_config` on this flow passes the model, API base, and API key, and does not pass `max_tokens`.

## Filter flow

```text
keep_consistent     ColumnValueFilterBlock    quality_consistent eq true
keep_high_quality   ColumnValueFilterBlock    quality_score ge 0.8
```

`filter_value` may be a single value. The block wraps it in a list. The runner passes the caller's threshold with `runtime_params` on `keep_high_quality`. The YAML default remains 0.8.

Rows dropped by either block are saved with `stage: quality` and `quality_reason`. The runner detects them by `candidate_id`, because the block does not return a rejection record. If the filter raises `EmptyDatasetError`, every graded row is rejected and the run still writes the rejection file.

Set quality review off to keep every row that passes structural validation. Those rows are marked `quality_checked: false`. The review and filter flows are not loaded.

## Outputs

These files are replaced on every run. Writes go to a temporary file and then replace the destination, so a crash does not leave a half-written output. The output path must not be the examples file or the schema file.

`output/tool_calling_dataset_generated.jsonl` is the accepted records followed by the rejected records. Each line is the same object written to `output/tool_calling_dataset_synthetic.jsonl` or `output/tool_calling_dataset_synthetic.rejected.jsonl`. `generated` in the run report is this count, so it matches `requested`. Accepted rows go to `output/tool_calling_dataset_synthetic.jsonl`. Each accepted line includes:

- `messages`
- `source_question`
- `iterations` (count of assistant messages)
- `mode`: `offline_mcp_synthesis`
- `teacher_model`
- `synthetic_tool_results`: `true`
- `tools` (the schema used)
- `seed_query`
- `variant` (1-based index within the seed)
- `quality_checked`

Reviewed rows also include `quality_score`, `quality_consistent`, `quality_reason`, and `quality_reviewer_model`.

Rejected rows go to `output/tool_calling_dataset_synthetic.rejected.jsonl` with the stage, the error or reason, and either the raw response or the candidate record.

The run report is:

```json
{
  "requested": 0,
  "accepted": 0,
  "rejected": 0,
  "generated": 0,
  "quality_check": true,
  "min_quality_score": 0.8,
  "output": "output/tool_calling_dataset_synthetic.jsonl",
  "generated_output": "output/tool_calling_dataset_generated.jsonl",
  "rejected_output": "output/tool_calling_dataset_synthetic.rejected.jsonl"
}
```

`requested` is the number of generation calls. Weak rows are never promoted to fill that count.

## Runtime configuration

| Setting | Where it is set | Default |
| --- | --- | --- |
| Teacher model, API base, API key | `set_model_config` from `TEACHER_MODEL_NAME` or `MODEL`, `TEACHER_MODEL_URL` or `API_BASE`, `TEACHER_API_KEY` or `API_KEY` or `OPENAI_API_KEY` | required to run |
| `variants_per_seed` | `runtime_params` on `expand_seeds.num_samples` | 2 |
| Generation temperature | `generate_conversation` in `generation/flow.yaml` | 0.7 |
| Generation `max_tokens` | `set_model_config` on the generation flow | 6000 |
| Review temperature and `max_tokens` | `review_conversation` in `review/flow.yaml` | 0.0 and 1500 |
| Quality review | runner flag | on |
| Minimum score | `runtime_params` on `keep_high_quality.filter_value` | 0.8 |

The model id uses LiteLLM form, for example `openai/<served-model-name>` for an OpenAI-compatible endpoint. If no model is configured, the run is skipped.

Before a full run, `flow.validate_dataset` checks the required columns, and `flow.dry_run` checks that each YAML loads and the block chain accepts a one-row sample. Tests mock the teacher completion. They load each flow, assert the metadata id, accept a valid frame, and reject a frame that is missing a required column.

## What this flow does not claim

Review is a consistency heuristic. It can miss contradictions in invented data. Passing rows are simulated training text. They should be read before they are used for training. No shipment, inventory change, or order lookup has occurred.
