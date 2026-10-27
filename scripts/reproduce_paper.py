import argparse
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]

POLICY_LABELS = {
    "online_lp": "Algorithm 1",
    "round_robin": "Round Robin",
    "lor": "LOR",
    "random": "Random",
    "power_of_two": "Power-of-2",
}
FIGURE_POLICIES = ["online_lp", "round_robin", "lor", "random"]
TABLE_POLICIES = ["round_robin", "lor", "power_of_two", "online_lp"]

# (alpha, beta, gamma, sigma, zeta1, zeta2) as printed in the figure captions of the paper
WEIGHTS = {
    "avg_latency": (0.0, 1.0, 1.0, 0.0, 0.0, 0.0),
    "tail": (0.0, 0.0, 0.0, 0.0, 1.0, 1.0),
    "throughput": (1.0, 0.0, 0.0, 1.0, 0.0, 0.0),
    "avg_mix": (1 / 135, 1.0, 1.0, 1.0, 1 / 400, 1 / 400),
    "tail_mix": (1 / 135, 1 / 400, 1 / 400, 1.0, 1.0, 1.0),
    "throughput_mix": (1.0, 1 / 400, 1 / 400, 1.0, 1 / 400, 1 / 400),
    "fig2b": (1 / 27, 1 / 2048, 1 / 2048, 5.0, 1.0, 1.0),
}

# figure id -> (data, lambda, weights, noisy prediction)
FIGURES = {
    "2a": ("real", 0.4, "tail_mix", False),
    "2b": ("real", 0.4, "fig2b", True),
    "3a": ("real", 0.5, "avg_latency", False),
    "3b": ("real", 0.4, "tail", True),
    "4a": ("real", 0.4, "avg_mix", False),
    "4b": ("real", 0.4, "avg_latency", False),
    "5a": ("real", 0.4, "tail_mix", False),
    "5b": ("real", 0.4, "tail", False),
    "6a": ("real", 0.4, "throughput_mix", False),
    "6b": ("real", 0.4, "throughput", False),
    "7a": ("real", 0.5, "avg_mix", False),
    "7b": ("real", 0.5, "avg_latency", False),
    "8a": ("real", 0.5, "tail_mix", False),
    "8b": ("real", 0.5, "tail", False),
    "9a": ("real", 0.5, "throughput_mix", False),
    "9b": ("real", 0.5, "throughput", False),
    "10a": ("real", 0.4, "avg_mix", True),
    "10b": ("real", 0.4, "avg_latency", True),
    "11a": ("real", 0.4, "tail_mix", True),
    "11b": ("real", 0.4, "tail", True),
    "12a": ("real", 0.4, "throughput_mix", True),
    "12b": ("real", 0.4, "throughput", True),
    "14a": ("real", 0.5, "avg_mix", True),
    "14b": ("real", 0.5, "avg_latency", True),
    "15a": ("real", 0.5, "tail_mix", True),
    "15b": ("real", 0.5, "tail", True),
    "16a": ("real", 0.5, "throughput_mix", True),
    "16b": ("real", 0.5, "throughput", True),
    "17a": ("pd_1_4", 0.4, "avg_mix", False),
    "17b": ("pd_1_4", 0.4, "avg_latency", False),
    "18a": ("pd_1_4", 0.4, "tail_mix", False),
    "18b": ("pd_1_4", 0.5, "tail", False),
    "19a": ("pd_1_4", 0.4, "avg_mix", True),
    "19b": ("pd_1_4", 0.4, "avg_latency", True),
    "20a": ("pd_1_4", 0.4, "tail_mix", True),
    "20b": ("pd_1_4", 0.5, "tail", True),
    "21a": ("pd_4_1", 0.4, "avg_mix", False),
    "21b": ("pd_4_1", 0.4, "avg_latency", False),
    "22a": ("pd_4_1", 0.4, "tail_mix", False),
    "22b": ("pd_4_1", 0.5, "tail", False),
    "23a": ("pd_4_1", 0.4, "avg_mix", True),
    "23b": ("pd_4_1", 0.4, "avg_latency", True),
    "24a": ("pd_4_1", 0.4, "tail_mix", True),
    "24b": ("pd_4_1", 0.5, "tail", True),
}

# The paper does not state the setting of Tables 1-2; we use the noisy main-text setting (Fig. 2b).
TABLES = {
    "1": ("real", 0.4, "fig2b", True, False),
    "2": ("real", 0.4, "fig2b", True, True),
}
FIG13 = ("real", 0.4, "tail_mix", False)
FIG13_ZETA2 = [round(0.25 * i, 2) for i in range(21)]

UNIT_COLUMNS = {
    "words": ("input", "output"),
    "tokens": ("input_tokens", "output_tokens"),
}

PLOT_METRICS = [
    ("Thr", "throughput_tps"),
    ("EEL", "avg_eel"),
    ("TTFT", "avg_ttft"),
    ("QPS", "qps"),
    ("Tail EEL\n(99%)", "p99_eel"),
    ("Tail TTFT\n(99%)", "p99_ttft"),
]
BAR_STYLE = {
    "online_lp": {"color": "#e05a5a", "hatch": "////"},
    "round_robin": {"color": "#7aa6d6"},
    "lor": {"color": "#7fbf7f"},
    "random": {"color": "#c8a98e"},
    "power_of_two": {"color": "#b39ddb"},
}
TABLE_METRICS = [
    ("Avg EEL", "avg_eel", "down"),
    ("P95 EEL", "p95_eel", "down"),
    ("P99 EEL", "p99_eel", "down"),
    ("Thr.", "throughput_tps", "up"),
    ("SLO viol.", "slo_violation", "down"),
]
# relative improvement (%) over Round Robin reported in Tables 1 and 2 of the paper
PAPER_TABLES = {
    "table1": {
        "LOR": [0.67, 4.01, 6.19, 0.53, 0.98],
        "Power-of-2": [1.30, 3.84, 6.06, 0.73, 1.84],
        "Ours": [45.75, 42.49, 25.85, 0.90, 11.10],
    },
    "table2": {
        "LOR": [1.47, 6.72, 9.67, 1.48, 3.92],
        "Power-of-2": [1.80, 6.35, 8.28, 1.39, 3.28],
        "Ours": [44.81, 45.69, 31.15, 1.93, 14.57],
    },
}
LOWER_IS_BETTER = {"avg_eel", "avg_ttft", "p99_eel", "p99_ttft"}


@dataclass(frozen=True)
class RunSpec:
    data: str
    lam: float
    policy: str
    weights: Optional[str] = None
    noisy: bool = False
    rate_shift: bool = False
    zeta2: Optional[float] = None
    seed: int = 42

    @property
    def key(self) -> str:
        parts = [self.data, f"lam{self.lam:g}", self.policy]
        if self.policy == "online_lp":
            parts += [self.weights, "noisy" if self.noisy else "exact"]
            if self.zeta2 is not None:
                parts.append(f"zeta2_{self.zeta2:g}")
        if self.rate_shift:
            parts.append("rate_shift")
        parts.append(f"seed{self.seed}")
        return "__".join(parts)


def make_spec(data, lam, policy, weights, noisy, seed, rate_shift=False, zeta2=None):
    if policy != "online_lp":
        return RunSpec(data, lam, policy, rate_shift=rate_shift, seed=seed)
    return RunSpec(data, lam, policy, weights, noisy, rate_shift, zeta2, seed)


@dataclass
class Trace:
    path: str
    prefill_column: str
    decode_column: str
    mean_decode: float
    p99_decode: float
    mean_prefill: float


def describe_trace(df, prefill_column, decode_column, num_requests, path) -> Trace:
    used = df.head(num_requests)
    decode = used[decode_column].clip(lower=1)
    return Trace(
        path=str(path),
        prefill_column=prefill_column,
        decode_column=decode_column,
        mean_decode=float(decode.mean()),
        p99_decode=float(decode.quantile(0.99)),
        mean_prefill=float(used[prefill_column].clip(lower=1).mean()),
    )


def prepare_traces(args, output_root: Path) -> Dict[str, Trace]:
    length_csv = Path(args.length_csv).resolve()
    df = pd.read_csv(length_csv)
    prefill_column, decode_column = UNIT_COLUMNS[args.unit]
    traces = {
        "real": describe_trace(
            df, prefill_column, decode_column, args.num_requests, length_csv
        )
    }

    # synthetic P/D traces keep each real request's total length and re-split it with a fixed ratio
    total = (df[prefill_column] + df[decode_column]).to_numpy()
    trace_dir = output_root / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    for name, ratio in (("pd_1_4", 0.25), ("pd_4_1", 4.0)):
        prefill = np.maximum(1, np.round(total * ratio / (1.0 + ratio))).astype(int)
        decode = np.maximum(1, total - prefill).astype(int)
        synthetic = pd.DataFrame(
            {"num_prefill_tokens": prefill, "num_decode_tokens": decode}
        )
        path = trace_dir / f"{name}_{args.unit}.csv"
        synthetic.to_csv(path, index=False)
        traces[name] = describe_trace(
            synthetic, "num_prefill_tokens", "num_decode_tokens", args.num_requests, path
        )
    return traces


def time_step_for(args, lam: float) -> float:
    return float(args.periods.get(lam, args.time_step_seconds))


def qps_for(args, lam: float, time_step: float) -> float:
    factor = args.qps_per_lambda or 1.0 / time_step
    return lam * factor


def horizon_for(args, time_step: float) -> int:
    return max(1, int(math.ceil(args.time_limit / time_step)))


def tail_windows(args, trace: Trace):
    t2 = int(args.t2)
    t1 = int(args.t1) if args.t1 is not None else int(t2 + math.ceil(trace.p99_decode))
    return t1, t2


def reward_normalizer_for(args, weights, time_step: float) -> float:
    if args.reward_normalizer != "auto":
        return float(args.reward_normalizer)
    alpha, beta, gamma, sigma, zeta1, zeta2 = weights
    horizon = horizon_for(args, time_step)
    return max(1.0, (alpha + beta + gamma) * horizon + sigma + zeta1 + zeta2)


def rate_shift_params(args, lam: float, time_step: float):
    # the period stays fixed during the run; only the arrival rate switches
    qps_before = qps_for(args, lam, time_step)
    shift_time = (args.num_requests / 2.0) / qps_before
    return qps_before, qps_for(args, args.rate_shift_to, time_step), shift_time


def build_command(spec: RunSpec, args, trace: Trace, output_dir: Path) -> List[str]:
    time_step = time_step_for(args, spec.lam)
    cmd = [
        sys.executable,
        "-m",
        "vidur.main",
        "--seed",
        str(spec.seed),
        "--time_limit",
        str(args.time_limit),
        "--cluster_config_num_replicas",
        str(args.num_replicas),
        "--replica_config_model_name",
        args.model,
        "--replica_config_device",
        args.device,
        "--synthetic_request_generator_config_seed",
        str(spec.seed),
        "--synthetic_request_generator_config_num_requests",
        str(args.num_requests),
        "--length_generator_config_type",
        "trace",
        "--trace_request_length_generator_config_trace_file",
        trace.path,
        "--trace_request_length_generator_config_prefill_column",
        trace.prefill_column,
        "--trace_request_length_generator_config_decode_column",
        trace.decode_column,
        "--no-trace_request_length_generator_config_shuffle",
        "--no-metrics_config_enable_chrome_trace",
        "--no-metrics_config_store_plots",
        "--metrics_config_store_request_timeline",
        "--metrics_config_output_dir",
        str(output_dir),
        "--global_scheduler_config_type",
        spec.policy,
    ]

    if spec.rate_shift:
        qps_before, qps_after, shift_time = rate_shift_params(args, spec.lam, time_step)
        cmd += [
            "--interval_generator_config_type",
            "poisson_rate_shift",
            "--poisson_rate_shift_request_interval_generator_config_qps",
            str(qps_before),
            "--poisson_rate_shift_request_interval_generator_config_qps_after_shift",
            str(qps_after),
            "--poisson_rate_shift_request_interval_generator_config_shift_time",
            str(shift_time),
        ]
    else:
        cmd += [
            "--poisson_request_interval_generator_config_qps",
            str(qps_for(args, spec.lam, time_step)),
        ]

    if args.batch_size_cap is not None:
        cmd += ["--sarathi_scheduler_config_batch_size_cap", str(args.batch_size_cap)]

    if spec.policy == "power_of_two":
        cmd += ["--power_of_two_global_scheduler_config_seed", str(spec.seed)]

    if spec.policy == "online_lp":
        weights = list(WEIGHTS[spec.weights])
        if spec.zeta2 is not None:
            weights[5] = spec.zeta2
        alpha, beta, gamma, sigma, zeta1, zeta2 = weights
        t1, t2 = tail_windows(args, trace)
        prefix = "--online_lp_global_scheduler_config_"
        options = {
            "policy": "dual_ahd",
            "horizon": horizon_for(args, time_step),
            "time_step_seconds": time_step,
            "total_jobs": args.total_jobs,
            "grad_steps": args.grad_steps,
            "step_size": args.step_size,
            "batch_size": args.batch_size,
            "mem_scale_factor": args.mem_scale_factor,
            "backlog_gain": args.backlog_gain,
            "backlog_slack": args.backlog_slack,
            "length_penalty": args.length_penalty,
            "reward_normalizer": reward_normalizer_for(args, weights, time_step),
            "max_waiting": args.max_waiting,
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "sigma": sigma,
            "tail_eel_weight": zeta1,
            "tail_eel_window": t1,
            "goodput_weight": zeta2,
            "goodput_window": t2,
            "decode_prediction_noise": args.noise if spec.noisy else 0.0,
            "decode_prediction_seed": spec.seed,
        }
        if spec.rate_shift and args.rate_window_seconds > 0:
            options["rate_window_seconds"] = args.rate_window_seconds
            options["rate_shift_z"] = args.rate_shift_z
        for name, value in options.items():
            cmd += [f"{prefix}{name}", repr(value) if isinstance(value, float) else str(value)]
        cmd.append(f"{prefix}export_decisions")

    return cmd


def find_run_dir(base_dir: Path) -> Optional[Path]:
    if not base_dir.exists():
        return None
    candidates = sorted(
        p for p in base_dir.iterdir() if (p / "request_timeline.csv").exists()
    )
    return candidates[-1] if candidates else None


def run_spec(spec: RunSpec, args, trace: Trace, output_root: Path) -> Path:
    base_dir = output_root / "runs" / spec.key
    cmd = build_command(spec, args, trace, base_dir)
    command_line = " ".join(cmd)

    existing = find_run_dir(base_dir)
    log_path = base_dir / "vidur.log"
    if existing and not args.force and log_path.exists():
        with open(log_path) as log:
            if log.readline().rstrip("\n") == command_line:
                return existing

    base_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("WANDB_MODE", "disabled")
    with open(log_path, "w") as log:
        log.write(command_line + "\n")
        log.flush()
        result = subprocess.run(
            cmd, cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT
        )
    run_dir = find_run_dir(base_dir)
    if result.returncode != 0 or run_dir is None:
        raise RuntimeError(f"{spec.key} failed, see {base_dir / 'vidur.log'}")
    return run_dir


def compute_metrics(run_dir: Path, args, trace: Trace, time_step: float) -> dict:
    timeline = pd.read_csv(run_dir / "request_timeline.csv")
    final_time = float(timeline["final_time"].iloc[0])
    arrived = timeline["arrived_at"]
    eel = timeline["completed_at"].fillna(final_time) - arrived
    ttft = timeline["first_token_at"].fillna(final_time) - arrived
    completed = int(timeline["completed"].sum())

    t1, t2 = tail_windows(args, trace)
    slo_ttft = (
        args.slo_ttft_seconds if args.slo_ttft_seconds is not None else t2 * time_step
    )
    slo_e2e = (
        args.slo_e2e_seconds if args.slo_e2e_seconds is not None else t1 * time_step
    )

    metrics = {
        "requests": len(timeline),
        "completed": completed,
        "completion_rate": completed / max(1, len(timeline)),
        "throughput_tps": timeline["decode_tokens_generated"].sum() / args.time_limit,
        "qps": completed / args.time_limit,
        "avg_eel": eel.mean(),
        "p95_eel": eel.quantile(0.95),
        "p99_eel": eel.quantile(0.99),
        "avg_ttft": ttft.mean(),
        "p95_ttft": ttft.quantile(0.95),
        "p99_ttft": ttft.quantile(0.99),
        "slo_violation": float(((ttft > slo_ttft) | (eel > slo_e2e)).mean()),
        "ttft_within_t2": float((ttft <= t2 * time_step).mean()),
        "time_step_seconds": time_step,
        "final_time": final_time,
        "run_dir": str(run_dir),
    }

    request_metrics = pd.read_csv(run_dir / "request_metrics.csv")
    tpot = request_metrics.loc[
        request_metrics["request_completed"] > 0,
        "decode_time_execution_plus_preemption_normalized",
    ]
    metrics["median_tpot"] = float(tpot.median()) if len(tpot) else float("nan")

    instance_path = run_dir / "online_lp_instance.json"
    if instance_path.exists():
        summary = json.loads(instance_path.read_text())
        metrics["online_objective"] = summary["summary"]["online_objective"]
        metrics["lp_waiting_at_end"] = summary["summary"]["num_waiting_at_end"]
        metrics["rate_shift_events"] = len(summary["rate_shift_events"])
    return metrics


def collect_specs(args) -> Dict[str, List[RunSpec]]:
    groups: Dict[str, List[RunSpec]] = {}
    for seed in args.seeds:
        for fig_id in args.figures:
            data, lam, weights, noisy = FIGURES[fig_id]
            groups.setdefault(f"fig{fig_id}", []).extend(
                make_spec(data, lam, policy, weights, noisy, seed)
                for policy in FIGURE_POLICIES
            )
        for table_id in args.tables:
            data, lam, weights, noisy, rate_shift = TABLES[table_id]
            groups.setdefault(f"table{table_id}", []).extend(
                make_spec(data, lam, policy, weights, noisy, seed, rate_shift)
                for policy in TABLE_POLICIES
            )
        if args.fig13:
            data, lam, weights, noisy = FIG13
            groups.setdefault("fig13", []).extend(
                make_spec(data, lam, "online_lp", weights, noisy, seed, zeta2=zeta2)
                for zeta2 in FIG13_ZETA2
            )
    return groups


def mean_over_seeds(results: Dict[RunSpec, dict], specs: List[RunSpec]) -> pd.DataFrame:
    rows = []
    for spec in specs:
        if spec in results:
            rows.append(
                {"policy": spec.policy, "zeta2": spec.zeta2, **results[spec]}
            )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    numeric = df.drop(columns=["run_dir"]).groupby(
        ["policy", "zeta2"], dropna=False
    ).mean(numeric_only=True)
    return numeric.reset_index()


def plot_figure(fig_id: str, df: pd.DataFrame, path: Path, caption: str):
    df = df.set_index("policy").loc[[p for p in FIGURE_POLICIES if p in set(df["policy"])]]
    values = df[[key for _, key in PLOT_METRICS]]
    normalized = values / values.max(axis=0).replace(0, np.nan)

    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    x = np.arange(len(PLOT_METRICS))
    width = 0.8 / len(normalized)
    for i, policy in enumerate(normalized.index):
        ax.bar(
            x + (i - (len(normalized) - 1) / 2) * width,
            normalized.loc[policy].fillna(0).to_numpy(),
            width,
            label=POLICY_LABELS[policy],
            edgecolor="black",
            linewidth=0.6,
            **BAR_STYLE[policy],
        )
    ax.set_xticks(x)
    ax.set_xticklabels([label for label, _ in PLOT_METRICS])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Normalized value")
    ax.set_title(f"Fig. {fig_id}: {caption}", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(ncol=len(normalized), fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return normalized


def relative_improvement_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.set_index("policy")
    baseline = df.loc["round_robin"]
    rows = []
    for policy in [p for p in TABLE_POLICIES if p != "round_robin" and p in df.index]:
        row = {"Method": POLICY_LABELS[policy] if policy != "online_lp" else "Ours"}
        for label, key, direction in TABLE_METRICS:
            base = baseline[key]
            value = df.loc[policy, key]
            if not base:
                row[label] = float("nan")
            elif direction == "down":
                row[label] = (base - value) / base * 100.0
            else:
                row[label] = (value - base) / base * 100.0
        rows.append(row)
    return pd.DataFrame(rows)


def to_markdown(df: pd.DataFrame) -> str:
    lines = [
        "| " + " | ".join(df.columns) + " |",
        "|" + "|".join("---" for _ in df.columns) + "|",
    ]
    for row in df.itertuples(index=False):
        cells = [f"{v:.2f}" if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def caption_for(fig_id: str, args) -> str:
    data, lam, weights, noisy = FIGURES[fig_id]
    w = ", ".join(f"{v:.4g}" for v in WEIGHTS[weights])
    pred = f"o_hat ~ Unif({1 - args.noise:g}o, {1 + args.noise:g}o)" if noisy else "o_hat = o"
    return f"{data}, lambda={lam}, (a,b,g,s,z1,z2)=({w}), {pred}"


def write_report(args, groups, results, traces, output_root: Path):
    figures_dir = output_root / "figures"
    tables_dir = output_root / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    all_rows = [
        {**asdict(spec), "key": spec.key, **metrics} for spec, metrics in results.items()
    ]
    pd.DataFrame(all_rows).sort_values("key").to_csv(output_root / "results.csv", index=False)

    figure_summary = []
    for group, specs in groups.items():
        df = mean_over_seeds(results, specs)
        if df.empty:
            continue
        if group.startswith("fig") and group != "fig13":
            fig_id = group[3:]
            normalized = plot_figure(
                fig_id, df, figures_dir / f"{group}.png", caption_for(fig_id, args)
            )
            normalized.to_csv(figures_dir / f"{group}_normalized.csv")
            df.to_csv(figures_dir / f"{group}_raw.csv", index=False)
            if "online_lp" in normalized.index:
                baselines = normalized.drop(index="online_lp")
                row = {"figure": fig_id, "setting": caption_for(fig_id, args)}
                for label, key in PLOT_METRICS:
                    ours = normalized.loc["online_lp", key]
                    best = (
                        baselines[key].min()
                        if key in LOWER_IS_BETTER
                        else baselines[key].max()
                    )
                    row[label.replace("\n", " ")] = round(float(ours), 3)
                    row[f"best baseline {label.replace(chr(10), ' ')}"] = round(float(best), 3)
                    if key in LOWER_IS_BETTER:
                        row[f"wins {label.replace(chr(10), ' ')}"] = bool(ours < best)
                figure_summary.append(row)
        elif group == "fig13":
            df = df.sort_values("zeta2")
            fig, ax = plt.subplots(figsize=(6, 3.6))
            ax.plot(df["zeta2"], df["ttft_within_t2"] * 100.0, marker="o")
            ax.set_xlabel("zeta_2")
            ax.set_ylabel("Tail TTFT quantile (%)")
            ax.set_title(f"Fig. 13: share of requests with TTFT <= t'_2 = {args.t2} steps", fontsize=9)
            ax.grid(linestyle="--", alpha=0.4)
            fig.tight_layout()
            fig.savefig(figures_dir / "fig13.png", dpi=150)
            plt.close(fig)
            df.to_csv(figures_dir / "fig13_raw.csv", index=False)
        else:
            table = relative_improvement_table(df)
            table.to_csv(tables_dir / f"{group}.csv", index=False)
            (tables_dir / f"{group}.md").write_text(to_markdown(table))
            df.to_csv(tables_dir / f"{group}_raw.csv", index=False)
            paper = PAPER_TABLES.get(group)
            if paper:
                comparison = table.copy()
                for i, (label, _, _) in enumerate(TABLE_METRICS):
                    comparison[f"{label} (paper)"] = [
                        paper.get(method, [float("nan")] * len(TABLE_METRICS))[i]
                        for method in comparison["Method"]
                    ]
                ordered = ["Method"] + [
                    col
                    for label, _, _ in TABLE_METRICS
                    for col in (label, f"{label} (paper)")
                ]
                (tables_dir / f"{group}_vs_paper.md").write_text(
                    to_markdown(comparison[ordered])
                )

    if figure_summary:
        pd.DataFrame(figure_summary).to_csv(figures_dir / "summary.csv", index=False)


def vidur_batch_capacity(args) -> int:
    sys.path.insert(0, str(REPO_ROOT))
    from vidur.config import ReplicaConfig, SyntheticRequestGeneratorConfig
    from vidur.entities.replica import Replica
    from vidur.scheduler.utils.memory_planner import MemoryPlanner

    replica_config = ReplicaConfig(model_name=args.model, device=args.device)
    replica = Replica(replica_config, SyntheticRequestGeneratorConfig())
    batch_size_cap = args.batch_size_cap if args.batch_size_cap is not None else 128
    return min(MemoryPlanner(replica_config, replica).get_max_batch_size(), batch_size_cap)


def write_manifest(args, traces, output_root: Path, batch_capacity: int):
    lambdas = sorted(
        {FIGURES[f][1] for f in args.figures}
        | {TABLES[t][1] for t in args.tables}
        | ({FIG13[1]} if args.fig13 else set())
    )
    manifest = {
        "args": {**vars(args), "periods": {str(k): v for k, v in args.periods.items()}},
        "per_lambda": {
            str(lam): {
                "time_step_seconds": time_step_for(args, lam),
                "qps": qps_for(args, lam, time_step_for(args, lam)),
                "horizon": horizon_for(args, time_step_for(args, lam)),
            }
            for lam in lambdas
        },
        "batch_capacity_per_replica": batch_capacity,
        "traces": {},
        "assumptions": [
            "lambda is the arrival rate per period, qps = lambda * qps_per_lambda (default 1 / time_step_seconds)",
            "model and device are Vidur defaults; the paper only states four A100 GPUs",
            "t'_2 = 49 steps (Fig. 13); t'_1 = t'_2 + P99 decode length of the trace unless --t1 is given",
            "SLO violation: TTFT > t'_2 * time_step or EEL > t'_1 * time_step, with censored latencies for unfinished requests",
            "Tables 1-2 use the Fig. 2b setting (lambda 0.4, noisy prediction); Table 2 switches lambda at the arrival midpoint",
            "synthetic P/D traces re-split each real request's total length with the fixed ratio",
            "Fig. 13 sweeps zeta_2 on the Fig. 2a setting",
        ],
    }
    for name, trace in traces.items():
        t1, t2 = tail_windows(args, trace)
        manifest["traces"][name] = {
            **asdict(trace),
            "t1": t1,
            "t2": t2,
            "capacity_lambda_batch_bound": args.num_replicas
            * batch_capacity
            / trace.mean_decode,
        }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def run_all(args):
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    traces = prepare_traces(args, output_root)
    manifest = write_manifest(args, traces, output_root, vidur_batch_capacity(args))
    for name, info in manifest["traces"].items():
        print(
            f"{name}: mean decode {info['mean_decode']:.1f}, t1={info['t1']}, t2={info['t2']}, "
            f"batch-bound capacity lambda={info['capacity_lambda_batch_bound']:.3f}/period"
        )

    groups = collect_specs(args)
    unique_specs = sorted({spec for specs in groups.values() for spec in specs}, key=lambda s: s.key)
    print(f"{len(unique_specs)} unique runs for {len(groups)} figures/tables")

    results: Dict[RunSpec, dict] = {}
    failures = {}
    if args.report_only:
        for spec in unique_specs:
            run_dir = find_run_dir(output_root / "runs" / spec.key)
            if run_dir:
                results[spec] = compute_metrics(
                    run_dir, args, traces[spec.data], time_step_for(args, spec.lam)
                )
    else:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {
                executor.submit(run_spec, spec, args, traces[spec.data], output_root): spec
                for spec in unique_specs
            }
            for done, future in enumerate(as_completed(futures), start=1):
                spec = futures[future]
                try:
                    run_dir = future.result()
                    results[spec] = compute_metrics(
                        run_dir, args, traces[spec.data], time_step_for(args, spec.lam)
                    )
                    print(f"[{done}/{len(unique_specs)}] {spec.key}")
                except Exception as exc:
                    failures[spec.key] = str(exc)
                    print(f"[{done}/{len(unique_specs)}] FAILED {spec.key}: {exc}")

    write_report(args, groups, results, traces, output_root)
    print(f"Report written to {output_root}")
    if failures:
        print(f"{len(failures)} runs failed")
        sys.exit(1)


def calibrate(args):
    output_root = Path(args.output_root).resolve() / "calibration"
    output_root.mkdir(parents=True, exist_ok=True)
    traces = prepare_traces(args, output_root)
    trace = traces["real"]
    batch_capacity = vidur_batch_capacity(args)
    print(
        f"B per replica = {batch_capacity}, mean decode = {trace.mean_decode:.1f}, "
        f"batch-bound capacity = {args.num_replicas * batch_capacity / trace.mean_decode:.3f} requests/period"
    )
    env = os.environ.copy()
    env.setdefault("WANDB_MODE", "disabled")
    periods = {}
    for lam in args.lambdas:
        # f(dt) = TPOT(lambda / dt) - dt decreases with dt, so bisect for its root
        low, high = args.calibration_bounds
        for iteration in range(args.iterations):
            time_step = math.sqrt(low * high)
            step_args = argparse.Namespace(
                **{**vars(args), "time_step_seconds": time_step, "periods": {}}
            )
            spec = RunSpec("real", lam, "round_robin", seed=args.seeds[0])
            base_dir = output_root / f"lam{lam:g}_dt{time_step:.6f}"
            base_dir.mkdir(parents=True, exist_ok=True)
            if find_run_dir(base_dir) is None or args.force:
                with open(base_dir / "vidur.log", "w") as log:
                    subprocess.run(
                        build_command(spec, step_args, trace, base_dir),
                        cwd=REPO_ROOT,
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
            metrics = compute_metrics(find_run_dir(base_dir), step_args, trace, time_step)
            tpot = metrics["median_tpot"]
            print(
                f"lambda={lam} iter={iteration} dt={time_step:.5f}s qps={qps_for(step_args, lam, time_step):.2f} "
                f"-> median TPOT={tpot:.5f}s, completion={metrics['completion_rate']:.3f}, "
                f"avg TTFT={metrics['avg_ttft']:.3f}s"
            )
            if tpot > time_step:
                low = time_step
            else:
                high = time_step
        periods[lam] = math.sqrt(low * high)
        print(f"lambda={lam}: period = {periods[lam]:.5f}s (qps = {lam / periods[lam]:.2f})")
    (output_root / "periods.json").write_text(json.dumps(periods, indent=2))


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce the experiments of 'Online Linear Programming for Multi-Objective Routing in LLM Serving' in Vidur."
    )
    parser.add_argument("command", choices=["run", "calibrate"])
    parser.add_argument("--length-csv", required=True)
    parser.add_argument("--unit", choices=sorted(UNIT_COLUMNS), default="words")
    parser.add_argument("--num-requests", type=int, default=1254)
    parser.add_argument("--time-limit", type=int, default=180)
    parser.add_argument("--num-replicas", type=int, default=4)
    parser.add_argument("--model", default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--device", default="a100")
    parser.add_argument("--time-step-seconds", type=float, default=0.03)
    parser.add_argument(
        "--periods-json",
        default="",
        help="JSON mapping lambda -> period length in seconds (written by the calibrate command).",
    )
    parser.add_argument(
        "--qps-per-lambda",
        type=float,
        default=None,
        help="Arrivals per second for lambda = 1. Default: 1 / time_step_seconds (lambda per period).",
    )
    parser.add_argument(
        "--batch-size-cap",
        type=int,
        default=None,
        help="Replica (Sarathi) batch size cap. Default keeps Vidur's 128; the Online LP always uses the memory-planner bound.",
    )
    parser.add_argument("--noise", type=float, default=0.2)
    parser.add_argument("--t2", type=int, default=49)
    parser.add_argument("--t1", type=int, default=None)
    parser.add_argument("--slo-ttft-seconds", type=float, default=None)
    parser.add_argument("--slo-e2e-seconds", type=float, default=None)
    parser.add_argument("--total-jobs", type=int, default=1000)
    parser.add_argument("--grad-steps", type=int, default=5)
    parser.add_argument("--step-size", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--mem-scale-factor", type=float, default=1.0)
    parser.add_argument("--backlog-gain", type=float, default=1.0)
    parser.add_argument("--backlog-slack", type=float, default=1.0)
    parser.add_argument("--length-penalty", type=float, default=0.0)
    parser.add_argument("--max-waiting", type=int, default=0)
    parser.add_argument(
        "--reward-normalizer",
        default="1.0",
        help="Float, or 'auto' for (alpha + beta + gamma) * T + sigma + zeta1 + zeta2.",
    )
    parser.add_argument("--rate-shift-to", type=float, default=0.6)
    parser.add_argument("--rate-window-seconds", type=float, default=10.0)
    parser.add_argument("--rate-shift-z", type=float, default=3.0)
    parser.add_argument("--figures", nargs="*", default=[])
    parser.add_argument("--tables", nargs="*", default=[], choices=sorted(TABLES))
    parser.add_argument("--fig13", action="store_true")
    parser.add_argument("--lambdas", type=float, nargs="+", default=[0.4, 0.5])
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument(
        "--calibration-bounds",
        type=float,
        nargs=2,
        default=[0.005, 0.1],
        help="Bracket (seconds) for the period length bisection.",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--max-workers", type=int, default=max(1, min(6, (os.cpu_count() or 2) // 2)))
    parser.add_argument(
        "--output-root",
        default="",
        help="Default: simulator_output/paper_repro_<unit>.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()

    if not args.output_root:
        args.output_root = f"simulator_output/paper_repro_{args.unit}"

    args.periods = {}
    if args.periods_json:
        args.periods = {
            float(lam): float(period)
            for lam, period in json.loads(Path(args.periods_json).read_text()).items()
        }

    if args.figures == ["all"]:
        args.figures = list(FIGURES)
    unknown = [fig for fig in args.figures if fig not in FIGURES]
    if unknown:
        parser.error(f"unknown figures: {unknown}")

    if args.command == "calibrate":
        calibrate(args)
    else:
        run_all(args)


if __name__ == "__main__":
    main()
