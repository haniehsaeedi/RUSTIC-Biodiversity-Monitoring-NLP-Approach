"""
stats_tracker.py — Run statistics tracking for the RUSTIC pipeline.

Tracks every outcome for every document touched in a run, persists cumulative
totals across multiple days/sessions to a JSON file, and writes a per-run log
entry to a CSV history file for easy analysis later (e.g. in Excel or pandas).

Files written
-------------
output/pipeline_stats.json   — cumulative totals across ALL runs ever (resumable)
output/pipeline_runs.csv     — one row per run, for tracking progress over time
"""

import json
import os
import time
from collections import Counter
from datetime import datetime, timezone


class StatsTracker:
    """
    Tracks counts of every outcome a document can have during a pipeline run,
    plus timing. Call .record(reason) for every document processed. Call
    .finalize() at the end of the run to persist and print a summary.
    """

    # All possible outcomes — keeps reporting consistent even if a category
    # has zero occurrences in a given run.
    OUTCOMES = [
        "no_resource_link",     # OpenAlex had no PDF or landing page URL
        "download_failed",      # Request/timeout/connection error
        "boilerplate_detected", # Publisher landing page junk, paywall, etc.
        "empty_text",           # Downloaded but no usable text extracted
        "below_threshold",      # Scored but combined score < COMBO_THRESHOLD
        "matched",              # Scored and combined score >= COMBO_THRESHOLD
    ]

    def __init__(self, stats_json_path: str, runs_csv_path: str):
        self.stats_json_path = stats_json_path
        self.runs_csv_path   = runs_csv_path

        self.run_start_time  = time.time()
        self.run_start_iso   = datetime.now(timezone.utc).isoformat()

        # Per-run counters (reset each run)
        self.run_counts: Counter = Counter()

        # Cumulative counters (loaded from disk, persist across runs)
        self.cumulative = self._load_cumulative()

    # ── Recording ──────────────────────────────────────────────────────────

    def record(self, outcome: str) -> None:
        """Record one document's outcome. outcome must be one of self.OUTCOMES."""
        if outcome not in self.OUTCOMES:
            raise ValueError(f"Unknown outcome '{outcome}'. Must be one of {self.OUTCOMES}")
        self.run_counts[outcome] += 1

    # ── Derived metrics ────────────────────────────────────────────────────

    @property
    def total_touched_this_run(self) -> int:
        """Every document that entered the scoring funnel this run (any outcome)."""
        return sum(self.run_counts.values())

    @property
    def elapsed_seconds(self) -> float:
        return time.time() - self.run_start_time

    @property
    def docs_per_minute(self) -> float:
        mins = self.elapsed_seconds / 60
        return self.total_touched_this_run / mins if mins > 0 else 0.0

    @property
    def match_rate_pct(self) -> float:
        """% of touched documents that matched, this run."""
        touched = self.total_touched_this_run
        return (self.run_counts["matched"] / touched * 100) if touched else 0.0

    # ── Persistence ────────────────────────────────────────────────────────

    def _load_cumulative(self) -> dict:
        if not os.path.exists(self.stats_json_path):
            return {k: 0 for k in self.OUTCOMES} | {
                "total_runs": 0,
                "total_elapsed_seconds": 0.0,
                "first_run_started": self.run_start_iso,
            }
        with open(self.stats_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Ensure all outcome keys exist even if OUTCOMES list grows later
        for k in self.OUTCOMES:
            data.setdefault(k, 0)
        data.setdefault("total_runs", 0)
        data.setdefault("total_elapsed_seconds", 0.0)
        data.setdefault("first_run_started", self.run_start_iso)
        return data

    def _format_duration(self, seconds: float) -> str:
        h, rem = divmod(int(seconds), 3600)
        m, s   = divmod(rem, 60)
        return f"{h}h {m}m {s}s"

    def finalize(self, openalex_total_results: int | None = None,
                seen_ids_total: int | None = None) -> dict:
        """
        Call once at the end of a run (success or graceful stop).
        Persists cumulative stats, appends a row to the run history CSV,
        and prints a full summary table to the console.
        Returns the summary dict for optional further use.
        """
        elapsed = self.elapsed_seconds

        # ── Update cumulative totals ─────────────────────────────────────────
        for k in self.OUTCOMES:
            self.cumulative[k] = self.cumulative.get(k, 0) + self.run_counts[k]
        self.cumulative["total_runs"] = self.cumulative.get("total_runs", 0) + 1
        self.cumulative["total_elapsed_seconds"] = (
            self.cumulative.get("total_elapsed_seconds", 0.0) + elapsed
        )
        self.cumulative["last_run_finished"] = datetime.now(timezone.utc).isoformat()
        if seen_ids_total is not None:
            self.cumulative["total_unique_ids_seen"] = seen_ids_total
        if openalex_total_results is not None:
            self.cumulative["openalex_query_total_results"] = openalex_total_results

        os.makedirs(os.path.dirname(self.stats_json_path) or ".", exist_ok=True)
        with open(self.stats_json_path, "w", encoding="utf-8") as f:
            json.dump(self.cumulative, f, indent=2)

        # ── Append run-history CSV row ───────────────────────────────────────
        is_new_file = not os.path.exists(self.runs_csv_path)
        with open(self.runs_csv_path, "a", encoding="utf-8") as f:
            if is_new_file:
                header = (
                    "run_started_utc,run_finished_utc,elapsed_seconds,elapsed_human,"
                    + ",".join(self.OUTCOMES)
                    + ",total_touched,match_rate_pct,docs_per_minute,"
                    "cumulative_total_runs,cumulative_unique_ids_seen\n"
                )
                f.write(header)
            row = (
                f"{self.run_start_iso},"
                f"{datetime.now(timezone.utc).isoformat()},"
                f"{elapsed:.1f},"
                f"{self._format_duration(elapsed)},"
                + ",".join(str(self.run_counts[k]) for k in self.OUTCOMES)
                + f",{self.total_touched_this_run},"
                f"{self.match_rate_pct:.2f},"
                f"{self.docs_per_minute:.2f},"
                f"{self.cumulative['total_runs']},"
                f"{seen_ids_total or ''}\n"
            )
            f.write(row)

        # ── Console summary ──────────────────────────────────────────────────
        self._print_summary(openalex_total_results, seen_ids_total)

        return {
            "this_run": dict(self.run_counts),
            "cumulative": self.cumulative,
            "elapsed_seconds": elapsed,
        }

    def _print_summary(self, openalex_total_results, seen_ids_total) -> None:
        c = self.run_counts
        elapsed = self.elapsed_seconds

        print(f"\n{'='*64}")
        print(f"  RUN SUMMARY")
        print(f"{'='*64}")
        print(f"  Duration this run        : {self._format_duration(elapsed)}")
        print(f"  Documents touched        : {self.total_touched_this_run:,}")
        print(f"  Processing rate          : {self.docs_per_minute:.1f} docs/min")
        print(f"  {'-'*60}")
        print(f"  Outcome breakdown (this run):")
        labels = {
            "no_resource_link":     "No PDF/landing page link",
            "download_failed":      "Download failed (timeout/error)",
            "boilerplate_detected": "Boilerplate / paywall page skipped",
            "empty_text":           "No usable text extracted",
            "below_threshold":      "Scored — below match threshold",
            "matched":              "Scored — RUSTIC MATCH ✅",
        }
        for k in self.OUTCOMES:
            n   = c[k]
            pct = n / self.total_touched_this_run * 100 if self.total_touched_this_run else 0
            print(f"    {labels[k]:<38} {n:>7,}  ({pct:5.1f}%)")
        print(f"  {'-'*60}")
        print(f"  Match rate (matched / touched) : {self.match_rate_pct:.2f}%")

        print(f"\n  CUMULATIVE TOTALS (all runs to date):")
        print(f"  Total runs                : {self.cumulative['total_runs']:,}")
        print(f"  Total elapsed time        : "
              f"{self._format_duration(self.cumulative['total_elapsed_seconds'])}")
        for k in self.OUTCOMES:
            print(f"    {labels[k]:<38} {self.cumulative[k]:>7,}")
        if seen_ids_total is not None:
            print(f"  Total unique IDs seen     : {seen_ids_total:,}")
        if openalex_total_results:
            completion_pct = seen_ids_total / openalex_total_results * 100 \
                if seen_ids_total else 0
            print(f"  OpenAlex query total      : {openalex_total_results:,}")
            print(f"  Query completion          : {completion_pct:.2f}%")
            if seen_ids_total and self.docs_per_minute > 0:
                remaining = openalex_total_results - seen_ids_total
                est_minutes = remaining / self.docs_per_minute
                print(f"  Estimated time remaining  : "
                      f"{self._format_duration(est_minutes * 60)} "
                      f"(at current rate)")
        print(f"{'='*64}\n")
