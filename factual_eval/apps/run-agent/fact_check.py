import asyncio
import logging
import pathlib
import json
import re
import os
import signal
from miroflow.llm.client import LLMClient
from pathlib import Path
import dotenv
import hydra
from miroflow.contrib.tracing import set_tracing_disabled, set_tracing_export_api_key
from miroflow.contrib.tracing.otlp_setup import bootstrap_silent_trace_provider
from miroflow.logging.logger import bootstrap_logger
from miroflow.prebuilt.config import config_name, config_path, debug_config
from miroflow.prebuilt.pipeline import (
    create_pipeline_components,
    execute_task_pipeline,
)
from omegaconf import DictConfig

def signal_handler(signum, frame):
    """Force exit signal handler"""
    print(f"\n⚠️  Received interrupt signal {signum}, forcing immediate exit...")
    print("Program will terminate all operations immediately")
    os._exit(1)  # Force immediate exit
def extract_json_block(s: str) -> str:
    # 1) 去掉 Markdown 围栏 ```...```
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s.strip(), flags=re.IGNORECASE | re.DOTALL)

    # 2) 截取从第一个 { 到最后一个 } 的子串
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON object braces found in input.")
    s = s[start:end+1]
    return s

def build_prompt(user_query, long_report):
    TASK = """"""
    return (
        TASK.strip()
        + "\n[user_query start]>>>\n"
        + user_query
        + "\n[user_query end]>>>\n"
        + "\n[part_start]>>>\n"
        + long_report
        + "\n[part_end]>>>\n"
    )

async def segment_report(task_description: str, llm_client) -> list[str]:

    system_prompt = """You are a text segmentation assistant. Your task is to segment reports into logical parts:
- Avoid single-sentence parts.
- Do NOT alter the original content.
- Titles, headings, or section headers should be combined with the following content as one part.
- If multiple sentences clearly belong to the same section, keep them together in one part instead of splitting them.
- Output must be ONLY a valid JSON list, where each element is one part.

Example output format:
[
  "This is the first part.",
  "This is the second part.",
  "This is the third part."
]"""

    user_message = f"Please segment the following report into logical parts:\n\n{task_description}"

    message_history = [
        {
            "role": "user",
            "content": user_message
        }
    ]

    response = await llm_client.create_message(
        system_prompt=system_prompt,
        message_history=message_history,
        tool_definitions=[],
        keep_tool_result=-1
    )

    # Extract the response text from the LLM response
    assistant_response_text, _ = llm_client.process_llm_response(
        response, message_history, "main"
    )

    # Ensure the result is a valid JSON list
    try:
        parsed_result = json.loads(assistant_response_text.strip())
        if isinstance(parsed_result, list) and all(isinstance(x, str) for x in parsed_result):
            return parsed_result
        else:
            raise ValueError("LLM did not return a valid list of strings")
    except Exception as e:
        raise RuntimeError(f"Failed to parse segmentation output: {e}\nRaw output: {assistant_response_text}")

async def single_task(
    cfg: DictConfig,
    logger: logging.Logger,
    task_id: str = "task_1",
    task_description: str = "Write a python code to say 'Hello, World!', use python to execute the code.",
    task_file_name: str = "",
    idx: int = 0,
    input_model: str = "",
) -> str:
    """Asynchrono us main function."""
    logs_dir = Path(cfg.output_dir)
    main_agent_tool_manager, sub_agent_tool_managers, output_formatter = (
        create_pipeline_components(cfg, logs_dir=str(logs_dir))
    )

    task_name = task_id
    if input_model:
        log_path = pathlib.Path(".") / pathlib.Path(cfg.output_dir) / input_model / f"{task_name}_{idx}.log"
    else:
        log_path = pathlib.Path(".") / pathlib.Path(cfg.output_dir) / f"{task_name}_{idx}.log"
    logger.info(f"logger_path is {log_path.absolute()}")

    # Execute task using the pipeline
    final_summary, final_boxed_answer, _ = await execute_task_pipeline(
        cfg=cfg,
        task_name=task_name,
        task_id=task_id,
        task_file_name=task_file_name,
        task_description=task_description,
        main_agent_tool_manager=main_agent_tool_manager,
        sub_agent_tool_managers=sub_agent_tool_managers,
        output_formatter=output_formatter,
        # relative to the folder where shell command is launched.
        log_path=log_path.absolute(),
    )

    # Print task result
    logger.info(
        f"Final Output for Task: {task_id}, summary = {final_summary}"
    )
    return final_summary

async def entrypoint(
    cfg: DictConfig,
    logger: logging.Logger,
    task_id: str = "task_1",
    task: str = "Write a python code to say 'Hello, World!', use python to execute the code.",
    user_query: str = "",
    task_file_name: str = "",
    input_model: str = "",
) -> None:
    debug_config(cfg, logger)
    llm_client = LLMClient(task_id=task_id, cfg=cfg)
    max_retries = 5
    for attempt in range(max_retries):
        try:
            segmented_description = await segment_report(task, llm_client)
            break
        except Exception as e:
            logger.warning(f"Segment report attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                logger.error(f"All {max_retries} attempts to segment report failed, returning None")
                return
            else:
                await asyncio.sleep(1)  # Wait 1 second before retry

    semaphore = asyncio.Semaphore(20)
    async def run_with_semaphore(idx, sub_task, user_query):
        task_description = build_prompt(user_query, sub_task)
        try:
            async with semaphore:
                for attempt in range(max_retries):
                    final_answer_text = await single_task(cfg, logger, task_id, task_description, task_file_name, idx, input_model)
                    if final_answer_text!="No final answer generated." and final_answer_text != "Unable to generate final summary due to persistent network issues. You should try again.":
                        break
                    if attempt < max_retries -1:
                        await asyncio.sleep(1)
                    if attempt == max_retries - 1:
                        return {"idx": idx, "ok": False, "error": "No final answer generated.", "origin": task_description,}
                return {"idx": idx, "ok": True, "origin": task_description,
                        "final_answer_text": final_answer_text}
        except Exception as e:
            logger.error(f"Failed to run task at idx {idx}: {e}")
            return {"idx": idx, "ok": False, "error": repr(e), "origin": task_description,}
    # Run tasks in parallel
    results = await asyncio.gather(
        *[run_with_semaphore(idx, sub_task, user_query) for idx, sub_task in enumerate(segmented_description)], return_exceptions=False
    )

    results.sort(key=lambda x: x["idx"])
    merged = []
    for r in results:
        if r["ok"]:
            try:
                parsed = json.loads(extract_json_block(r["final_answer_text"]))  # 每个是 JSON list
                # if isinstance(parsed, dict):
                if "core_state" in parsed and isinstance(parsed["core_state"], list):
                    merged.extend(parsed["core_state"])
                    logger.info(f"Task {r['idx']} success: {r['origin']}")
                # 兼容另一种可能的键名
                else:
                    # 字典但无上述键：作为一个元素加入
                    merged.append(parsed)

                # elif isinstance(parsed, list):
                #     merged.extend(parsed)
            except Exception as e:
                logger.error(f"Failed to parse result at idx {r['idx']}: {e}\nRaw: {r['final_answer_text']}")
        else:
            logger.error(f"Task {r['idx']} failed: {r['error']} origin test: {r['origin']}")
    logger.info(merged)
    output_dir = "../../results/"
    if input_model:
        output_dir = output_dir + input_model
    os.makedirs(output_dir, exist_ok=True)  # 确保目录存在

    output_path = os.path.join(output_dir, f"{task_id}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

def main(
    *args,
    task_id: str = "task_1",
    task: str = "Write a python code to say 'Hello, World!', use python to execute the code.",
    task_file_name: str = "",
    user_query: str = "",
    input_model: str = "",
):
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    dotenv.load_dotenv()
    with hydra.initialize_config_dir(config_dir=config_path(), version_base=None):
        cfg = hydra.compose(config_name=config_name(), overrides=list(args))
        logger = bootstrap_logger()
        # disable tracing and give a fake key
        set_tracing_disabled(True)
        set_tracing_export_api_key("fake-key")
        # suppress warning from trace_provider
        bootstrap_silent_trace_provider()
        asyncio.run(entrypoint(cfg, logger,task_id, task, user_query, task_file_name, input_model))

