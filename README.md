# offline-mcp-synthesis

Generate simulated tool-calling training conversations from sample transcripts, seed queries, a system prompt, and a tool schema. A teacher model writes the new questions and the tool results. This run does not start the sample MCP server, register tools with the model, or call an agent.

The behavior is specified in [DESIGN.md](DESIGN.md). The Order Ops files in this folder are one server pack. Replacing the pack is how you point the same flows at another MCP server.

## 1. Create a Python 3.12 environment

From this folder:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

`sdg_hub` is installed from the package name in `requirements.txt`. If you are working from the local SDG Hub checkout, install that checkout instead:

```bash
python -m pip install -e ../../sdg_hub
python -m pip install pandas python-dotenv jsonschema fastmcp
```

## 2. Configure the teacher

Create `.env` in this folder:

```dotenv
TEACHER_MODEL_NAME=openai/your-served-model
TEACHER_MODEL_URL=https://your-model-endpoint/v1
TEACHER_API_KEY=your-key
```

`TEACHER_MODEL_NAME` uses LiteLLM's provider/model form. For an OpenAI-compatible endpoint that is `openai/<served-model-name>`. `MODEL`, `API_BASE`, `API_KEY`, and `OPENAI_API_KEY` are accepted as fallbacks.

## 3. Review the Order Ops pack

These files are the inputs. The flows do not infer seeds from the samples.

| File | Role |
| --- | --- |
| `input/examples.jsonl` | Sample tool-calling conversations, including one inconsistent shipment answer the teacher is told not to copy |
| `input/seeds.json` | Seed queries |
| `input/system_prompt.txt` | System prompt copied into every generated transcript |
| `input/tool_schema.json` | OpenAI function schemas for `list_orders`, `get_order`, `check_inventory`, and `create_shipment` |
| `generation/prompts/synthesize.yaml` | Domain rules for writing a conversation |
| `review/prompts/review.yaml` | Domain rules for the consistency review |

`mcp_server.py` is a runnable FastMCP server with the same catalog. Synthesis does not connect to it. Start it only when you want to call the tools yourself:

```bash
python mcp_server.py
```

It listens on `http://127.0.0.1:8008/mcp`. `PORT` and `HOST` override the bind address.

## 4. Run the tests

The tests mock the teacher and do not call a model endpoint. They also exercise the Order Ops tool functions in-process.

```bash
python -m unittest discover -s tests -t . -v
```

## 5. Generate conversations

```bash
python offline_mcp_synthesis.py
```

The command reads `.env` and the files in `input/`, then writes:

- `output/tool_calling_dataset_generated.jsonl` for the accepted records followed by the rejected records
- `output/tool_calling_dataset_synthetic.jsonl` for rows that pass the filter
- `output/tool_calling_dataset_synthetic.rejected.jsonl` for rows rejected during structural checks, review parsing, or quality filtering

A run with zero accepted rows still exits successfully and still writes the rejection file. Read both files before using them for training. Tool results are simulated. The review score is a consistency heuristic.

Optional environment overrides:

| Variable | Default |
| --- | --- |
| `VARIANTS_PER_SEED` | `2` |
| `QUALITY_CHECK` | `true` |
| `MIN_QUALITY_SCORE` | `0.8` |
| `EXAMPLES_PATH` | `input/examples.jsonl` |
| `SEEDS_PATH` | `input/seeds.json` |
| `SYSTEM_PROMPT_PATH` | `input/system_prompt.txt` |
| `TOOL_SCHEMA_PATH` | `input/tool_schema.json` |
| `OUTPUT_PATH` | `output/tool_calling_dataset_synthetic.jsonl` |

Set `QUALITY_CHECK=false` to keep every row that passes structural validation and skip the review flow.

## 6. Use another MCP server

Replace the pack together. Leave `generation/flow.yaml`, `review/flow.yaml`, and `filter/flow.yaml` as they are.

1. Put the new function schemas in `input/tool_schema.json` (or point `TOOL_SCHEMA_PATH` at another file).
2. Replace `input/examples.jsonl` with transcripts that use those tools.
3. Replace `input/seeds.json` and `input/system_prompt.txt`.
4. Rewrite the domain sentences in `generation/prompts/synthesize.yaml` and `review/prompts/review.yaml`.
5. Replace `mcp_server.py` if you want a live server for inspection. The synthesizer still will not call it.

Then repeat steps 4 and 5.
