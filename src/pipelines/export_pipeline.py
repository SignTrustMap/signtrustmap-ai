import asyncio
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import httpx
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential
from src.api.schemas import ModelExportWebhookPayload

export_lock = asyncio.Lock()
export_tasks: Dict[str, Dict[str, Any]] = {}


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), reraise=False)
async def send_export_webhook(callback_url: str, payload: ModelExportWebhookPayload) -> None:
    logger.info(f"Delivering export webhook for task ID {payload.task_id} to {callback_url}")
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(callback_url, json=payload.model_dump())
        response.raise_for_status()
    logger.info(f"Export webhook delivered successfully for task ID {payload.task_id}")


async def run_export_pipeline(
    task_id: str,
    weights_path: str,
    export_format: str,
    half: bool = True,
    device: int = 0,
    workspace: int = 1,
    dynamic: bool = False,
    opset: int = 18,
    callback_url: Optional[str] = None,
) -> None:
    start_time = time.time()
    logger.info(f"Starting model export pipeline for task ID {task_id}: weights={weights_path}, format={export_format}")
    
    export_tasks[task_id] = {
        "task_id": task_id,
        "status": "PROCESSING",
        "weights_name": os.path.basename(weights_path),
        "format": export_format,
        "exported_files": [],
        "error": None,
        "duration_seconds": None,
    }

    base_dir = Path(__file__).resolve().parent.parent.parent
    script_path = base_dir / "scripts" / "export_model.py"

    cmd = [
        sys.executable,
        str(script_path),
        "--weights", str(weights_path),
        "--format", export_format,
        "--device", str(device),
        "--workspace", str(workspace),
        "--opset", str(opset),
    ]

    if half:
        cmd.append("--half")
    else:
        cmd.append("--no-half")

    if dynamic:
        cmd.append("--dynamic")

    error_logs = []
    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        async def read_stream(stream, is_stderr=False):
            while True:
                line = await stream.readline()
                if not line:
                    break
                decoded = line.decode("utf-8", errors="replace").rstrip()
                if decoded:
                    if is_stderr:
                        error_logs.append(decoded)
                        logger.warning(f"[{task_id}] {decoded}")
                    else:
                        logger.debug(f"[{task_id}] {decoded}")

        await asyncio.gather(
            read_stream(process.stdout, is_stderr=False),
            read_stream(process.stderr, is_stderr=True),
        )

        return_code = await process.wait()
        duration = round(time.time() - start_time, 2)

        if return_code == 0:
            exported_files = []
            weights_stem = Path(weights_path).stem
            weights_dir = Path(weights_path).parent

            if export_format in ("onnx", "all"):
                onnx_path = weights_dir / f"{weights_stem}.onnx"
                if onnx_path.exists():
                    exported_files.append(str(onnx_path))

            if export_format in ("engine", "all"):
                engine_path = weights_dir / f"{weights_stem}.engine"
                if engine_path.exists():
                    exported_files.append(str(engine_path))

            export_tasks[task_id]["status"] = "COMPLETED"
            export_tasks[task_id]["exported_files"] = exported_files
            export_tasks[task_id]["duration_seconds"] = duration
            logger.info(f"Model export task {task_id} completed successfully in {duration}s. Files: {exported_files}")

            if callback_url:
                payload = ModelExportWebhookPayload(
                    task_id=task_id,
                    status="COMPLETED",
                    exported_files=exported_files,
                    duration_seconds=duration,
                )
                await send_export_webhook(callback_url, payload)
        else:
            err_msg = "\n".join(error_logs[-5:]) if error_logs else f"Subprocess exited with code {return_code}"
            export_tasks[task_id]["status"] = "FAILED"
            export_tasks[task_id]["error"] = err_msg
            export_tasks[task_id]["duration_seconds"] = duration
            logger.error(f"Model export task {task_id} failed with code {return_code}: {err_msg}")

            if callback_url:
                payload = ModelExportWebhookPayload(
                    task_id=task_id,
                    status="FAILED",
                    error=err_msg,
                    duration_seconds=duration,
                )
                await send_export_webhook(callback_url, payload)

    except Exception as e:
        duration = round(time.time() - start_time, 2)
        err_msg = str(e)
        export_tasks[task_id]["status"] = "FAILED"
        export_tasks[task_id]["error"] = err_msg
        export_tasks[task_id]["duration_seconds"] = duration
        logger.error(f"Unexpected error in export pipeline task {task_id}: {err_msg}")

        if callback_url:
            payload = ModelExportWebhookPayload(
                task_id=task_id,
                status="FAILED",
                error=err_msg,
                duration_seconds=duration,
            )
            await send_export_webhook(callback_url, payload)

    finally:
        if export_lock.locked():
            export_lock.release()
            logger.debug("Released model export concurrency lock.")
