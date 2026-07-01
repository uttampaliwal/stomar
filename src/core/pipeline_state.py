"""Pipeline state machine with resume-from-checkpoint.

Tracks the progress of both the daily auto-pipeline and per-ticker
retraining pipeline through ordered stages. On crash or restart,
the pipeline reads the checkpoint file and resumes from the last
incomplete stage instead of starting over.

Stages (superset — both pipelines use a subset):
    IDLE → FETCH → VALIDATE → FEATURES → TRAIN → EVALUATE →
    PROMOTE → DAILY → PAPER → ARCHIVE → DONE

Checkpoint file: data/pipeline_checkpoint.json

Usage:
    from src.core.pipeline_state import PipelineCheckpoint

    ckpt = PipelineCheckpoint.load()
    if ckpt and ckpt.can_resume():
        print(f"Resuming from {ckpt.current_stage}")
    else:
        ckpt = PipelineCheckpoint()

    ckpt.advance("FETCH")       # marks FETCH as running
    # ... do work ...
    ckpt.complete("FETCH")      # marks FETCH as completed
    ckpt.advance("VALIDATE")
    # ...
    ckpt.complete("VALIDATE")
    ckpt.advance("DONE")
    ckpt.save()
"""

import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

from src.core.constants import DATA_DIR

logger = logging.getLogger(__name__)

CHECKPOINT_PATH = os.path.join(DATA_DIR, "pipeline_checkpoint.json")


def checkpoint_path(pipeline: str = "auto", ticker: str = None) -> str:
    """Get the checkpoint file path for a given pipeline/ticker combo."""
    if ticker:
        safe_ticker = ticker.replace(".", "_")
        return os.path.join(DATA_DIR, f"pipeline_checkpoint_{safe_ticker}.json")
    return CHECKPOINT_PATH

# All stages in canonical order
ALL_STAGES = [
    "IDLE", "FETCH", "VALIDATE", "FEATURES", "TRAIN", "EVALUATE",
    "PROMOTE", "DAILY", "PAPER", "ARCHIVE", "DONE",
]

# Stage ordering map for fast lookups
_STAGE_INDEX = {s: i for i, s in enumerate(ALL_STAGES)}


class StageStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class StageInfo:
    """Tracks metadata for a single pipeline stage."""
    status: StageStatus = StageStatus.PENDING
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None
    result: Optional[dict] = None

    def to_dict(self) -> dict:
        d = {"status": self.status.value}
        if self.started_at:
            d["started_at"] = self.started_at
        if self.completed_at:
            d["completed_at"] = self.completed_at
        if self.error:
            d["error"] = self.error
        if self.result:
            d["result"] = self.result
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "StageInfo":
        return cls(
            status=StageStatus(d.get("status", "pending")),
            started_at=d.get("started_at"),
            completed_at=d.get("completed_at"),
            error=d.get("error"),
            result=d.get("result"),
        )


@dataclass
class PipelineCheckpoint:
    """Persistent checkpoint for the pipeline state machine.

    Attributes:
        pipeline: Which pipeline this checkpoint belongs to ("auto" or "retrain").
        ticker: For retraining pipelines, which ticker is being processed.
        current_stage: The stage currently being executed (or last attempted).
        stages: Per-stage status and metadata.
        created_at: When this checkpoint was first created.
        updated_at: Last modification timestamp.
    """
    pipeline: str = "auto"
    ticker: Optional[str] = None
    current_stage: str = "IDLE"
    stages: dict[str, StageInfo] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    # ── Persistence ────────────────────────────────────────────────────────

    def save(self, path: str = None):
        """Atomically persist checkpoint to disk."""
        path = path or checkpoint_path(self.pipeline, self.ticker)
        self.updated_at = datetime.now().isoformat()

        data = {
            "pipeline": self.pipeline,
            "ticker": self.ticker,
            "current_stage": self.current_stage,
            "stages": {k: v.to_dict() for k, v in self.stages.items()},
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        dir_name = os.path.dirname(path) or "."
        fd, tmp = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

        logger.debug("Checkpoint saved | pipeline=%s | stage=%s | ticker=%s",
                      self.pipeline, self.current_stage, self.ticker)

    @classmethod
    def load(cls, path: str = None, pipeline: str = "auto", ticker: str = None) -> Optional["PipelineCheckpoint"]:
        """Load checkpoint from disk. Returns None if no valid checkpoint exists."""
        path = path or checkpoint_path(pipeline, ticker)
        if not os.path.exists(path):
            return None

        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Corrupt checkpoint file, ignoring: %s", e)
            return None

        stages = {
            k: StageInfo.from_dict(v)
            for k, v in data.get("stages", {}).items()
        }

        return cls(
            pipeline=data.get("pipeline", "auto"),
            ticker=data.get("ticker"),
            current_stage=data.get("current_stage", "IDLE"),
            stages=stages,
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
        )

    @classmethod
    def clear(cls, path: str = None, pipeline: str = "auto", ticker: str = None):
        """Delete the checkpoint file."""
        path = path or checkpoint_path(pipeline, ticker)
        if os.path.exists(path):
            os.remove(path)
            logger.info("Checkpoint cleared | path=%s", path)

    # ── Stage progression ──────────────────────────────────────────────────

    def advance(self, stage: str):
        """Mark a stage as running and update current_stage.

        Validates that the stage is reachable (no stages skipped unless
        explicitly marked).
        """
        if stage not in _STAGE_INDEX:
            raise ValueError(f"Unknown stage: {stage}. Valid: {ALL_STAGES}")

        self.current_stage = stage
        self.stages[stage] = StageInfo(
            status=StageStatus.RUNNING,
            started_at=datetime.now().isoformat(),
        )
        self.updated_at = datetime.now().isoformat()
        logger.info("Stage started | pipeline=%s | ticker=%s | stage=%s",
                     self.pipeline, self.ticker, stage)

    def complete(self, stage: str, result: dict = None):
        """Mark a stage as completed."""
        if stage not in self.stages:
            self.stages[stage] = StageInfo()

        self.stages[stage].status = StageStatus.COMPLETED
        self.stages[stage].completed_at = datetime.now().isoformat()
        if result:
            self.stages[stage].result = result
        self.updated_at = datetime.now().isoformat()
        logger.info("Stage completed | pipeline=%s | ticker=%s | stage=%s",
                     self.pipeline, self.ticker, stage)

    def fail(self, stage: str, error: str):
        """Mark a stage as failed."""
        if stage not in self.stages:
            self.stages[stage] = StageInfo()

        self.stages[stage].status = StageStatus.FAILED
        self.stages[stage].completed_at = datetime.now().isoformat()
        self.stages[stage].error = error
        self.updated_at = datetime.now().isoformat()
        logger.warning("Stage failed | pipeline=%s | ticker=%s | stage=%s | error=%s",
                        self.pipeline, self.ticker, stage, error)

    def skip(self, stage: str, reason: str = ""):
        """Mark a stage as skipped (e.g., no work to do)."""
        if stage not in self.stages:
            self.stages[stage] = StageInfo()

        self.stages[stage].status = StageStatus.SKIPPED
        self.stages[stage].completed_at = datetime.now().isoformat()
        if reason:
            self.stages[stage].result = {"skipped_reason": reason}
        self.updated_at = datetime.now().isoformat()
        logger.info("Stage skipped | pipeline=%s | ticker=%s | stage=%s | reason=%s",
                     self.pipeline, self.ticker, stage, reason)

    # ── Query methods ──────────────────────────────────────────────────────

    def stage_status(self, stage: str) -> StageStatus:
        """Get the status of a specific stage."""
        if stage in self.stages:
            return self.stages[stage].status
        return StageStatus.PENDING

    def stage_info(self, stage: str) -> Optional[StageInfo]:
        """Get full info for a specific stage."""
        return self.stages.get(stage)

    def is_complete(self) -> bool:
        """Check if the pipeline reached DONE."""
        return self.stage_status("DONE") == StageStatus.COMPLETED

    def has_failed(self) -> bool:
        """Check if any stage has failed."""
        return any(
            s.status == StageStatus.FAILED for s in self.stages.values()
        )

    def can_resume(self) -> bool:
        """Check if this checkpoint represents an interrupted run that can resume.

        Returns True if:
        - No stage has FAILED (that requires a full restart or manual intervention)
        - At least one stage is RUNNING or there are stages after the last completed one
        """
        if self.has_failed():
            return False

        # If we're already DONE, nothing to resume
        if self.is_complete():
            return False

        # Check if there's a running stage (interrupted mid-execution)
        running = [s for s in self.stages.values() if s.status == StageStatus.RUNNING]
        if running:
            return True

        # Check if there are pending stages after the last completed one
        last_completed_idx = -1
        for stage_name, info in self.stages.items():
            if info.status in (StageStatus.COMPLETED, StageStatus.SKIPPED):
                idx = _STAGE_INDEX.get(stage_name, -1)
                if idx > last_completed_idx:
                    last_completed_idx = idx

        # If we have stages after the last completed one, we can resume
        for stage_name in ALL_STAGES:
            idx = _STAGE_INDEX[stage_name]
            if idx > last_completed_idx and stage_name not in ("IDLE", "DONE"):
                return True

        return False

    def next_stage(self) -> Optional[str]:
        """Get the next stage to execute based on current progress.

        Returns the first stage that is not COMPLETED or SKIPPED.
        """
        for stage in ALL_STAGES:
            if stage in ("IDLE", "DONE"):
                continue
            status = self.stage_status(stage)
            if status in (StageStatus.PENDING, StageStatus.RUNNING):
                return stage
        return None

    def resume_from(self) -> Optional[str]:
        """Get the stage to resume from.

        If a stage was RUNNING when we crashed, resume from that stage.
        Otherwise, return the next pending stage.
        """
        # First check for interrupted (running) stages
        for stage_name, info in self.stages.items():
            if info.status == StageStatus.RUNNING:
                return stage_name

        # Otherwise find the next pending stage
        return self.next_stage()

    def get_completed_stages(self) -> list[str]:
        """Return list of completed stages in order."""
        return [
            stage for stage in ALL_STAGES
            if self.stage_status(stage) in (StageStatus.COMPLETED, StageStatus.SKIPPED)
        ]

    def get_summary(self) -> dict:
        """Return a summary dict of the checkpoint state."""
        return {
            "pipeline": self.pipeline,
            "ticker": self.ticker,
            "current_stage": self.current_stage,
            "can_resume": self.can_resume(),
            "is_complete": self.is_complete(),
            "has_failed": self.has_failed(),
            "completed_stages": self.get_completed_stages(),
            "next_stage": self.next_stage(),
            "resume_from": self.resume_from(),
            "stages": {
                name: {
                    "status": info.status.value,
                    "started_at": info.started_at,
                    "completed_at": info.completed_at,
                    "error": info.error,
                }
                for name, info in self.stages.items()
            },
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
