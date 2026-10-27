# Reproducing the Online LP routing paper

Scripts to reproduce the experiments of *Online Linear Programming for Multi-Objective Routing in LLM Serving* (Chen, Ye, Zhou, ICML 2026, arXiv:2607.03948) with this Vidur fork.

## Setup

```sh
uv venv --python 3.10 .venv
uv pip install --python .venv/bin/python -r requirements.txt huggingface_hub pyarrow tokenizers
```

## Data

LMSYS-Chat-1M is gated. Accept the terms at <https://huggingface.co/datasets/lmsys/lmsys-chat-1m>, create a read token and log in:

```sh
.venv/bin/hf auth login
.venv/bin/python scripts/prepare_lmsys_lengths.py
```

This writes `data/processed_traces/lmsys_chat_1m_first10000.csv` with lengths in words (`input`, `output`) and in Llama-2 tokens (`input_tokens`, `output_tokens`). It also writes `input_output_word_counts10000+(1).csv`, the file expected by `scripts/run_online_lp_policies.py --repro-input-output-csv`. Both files are git-ignored.

## Calibrating the period

The paper defines one period as one decode iteration. `calibrate` runs Round Robin and bisects the period length until it matches the median time per output token at each arrival rate:

```sh
.venv/bin/python scripts/reproduce_paper.py calibrate \
  --length-csv data/processed_traces/lmsys_chat_1m_first10000.csv --unit words
```

The result is written to `<output-root>/calibration/periods.json`.

## Running

```sh
.venv/bin/python scripts/reproduce_paper.py run \
  --length-csv data/processed_traces/lmsys_chat_1m_first10000.csv --unit words \
  --periods-json simulator_output/paper_repro_words/calibration/periods.json \
  --figures all --tables 1 2 --fig13
```

Outputs in `simulator_output/paper_repro_<unit>/`:

- `figures/figXX.png`: normalized bars as in the paper, with raw and normalized CSVs.
- `tables/table1.md`, `tables/table2.md`: relative improvement over Round Robin.
- `results.csv`: one row per run.
- `manifest.json`: parameters, derived rates and assumptions.
- `runs/<key>/<timestamp>/`: Vidur outputs, including `request_timeline.csv` and, for the Online LP, `online_lp_instance.json` and `online_lp_requests.csv`.

A run is reused only when its command line is unchanged. Use a different `--output-root` for each global setting you want to keep.

The Online LP bounds each replica by the memory-planner batch size (29 for Llama-2-7B on an A100 with 4096-token requests). Vidur's Sarathi scheduler admits up to `batch_size_cap = 128` requests instead. Pass `--batch-size-cap 29` (and calibrate with it) to make the simulated replicas match the router's model.

## New simulator options

All are off by default. With the defaults, outputs are identical to the original code.

| Option | Paper |
|---|---|
| `--global_scheduler_config_type power_of_two` | Power-of-2 Choices baseline |
| `--online_lp_global_scheduler_config_decode_prediction_noise 0.2` | o_hat ~ Unif(0.8 o, 1.2 o) |
| `--online_lp_global_scheduler_config_tail_eel_weight`, `..._tail_eel_window` | zeta_1, t'_1 |
| `--online_lp_global_scheduler_config_goodput_weight`, `..._goodput_window` (existing) | zeta_2, t'_2 |
| `--online_lp_global_scheduler_config_rate_window_seconds` | sliding-window rate detection (Table 2) |
| `--interval_generator_config_type poisson_rate_shift` | arrival-rate switch (Table 2) |
| `--online_lp_global_scheduler_config_export_decisions` | instance and decision export |
| `--metrics_config_store_request_timeline` | per-request timeline |

## Export format

`online_lp_instance.json` holds the horizon `T`, `time_step_seconds`, `batch_capacity` (`B_g`), `kv_capacity_tokens` (`M_g`), the reward weights and formula, the full Online LP config, and a summary with `online_objective`, the sum of raw rewards of the admitted requests.

`online_lp_requests.csv` has one row per request seen by the router: `request_id`, `arrived_at` (seconds), `t_arr` (step), `s`, `o_true`, `o_pred`, `started`, `replica_id`, `start_step`, `dispatched_at`, `reward`, `margin`.

An offline model over the same instance uses, for a start of request `j` on replica `g` at step `tau >= t_arr`:

- batch coefficient 1 at steps `tau .. tau + o_pred - 1`;
- memory coefficient `s + theta + 1` at step `tau + theta`, for `0 <= theta < o_pred`;
- reward `sigma + alpha*min(o, T-tau+1) + beta*max(0, T-(tau+o-1)) + gamma*max(0, T-tau+1) + zeta2*1[tau-t_arr < t2] + zeta1*1[tau+o-t_arr < t1]`, with `o = o_pred`.

Steps beyond `T` are not counted, as in the online code.

## Assumptions

The paper leaves these settings open. `manifest.json` records the values used in each run.

- `lambda` is the arrival rate per period: `qps = lambda / period`.
- Model and device are the Vidur defaults (Llama-2-7B, A100). The paper only states four A100 GPUs.
- `t'_2 = 49` steps (Fig. 13). `t'_1 = t'_2 + P99` decode length of the trace, unless `--t1` is given.
- SLO violation: TTFT above `t'_2` periods or EEL above `t'_1` periods. Unfinished requests use latencies censored at the end of the simulation.
- Tables 1 and 2 use the Fig. 2b setting. In Table 2 the rate switches once half of the requests are expected to have arrived.
- Synthetic P/D traces re-split each real request's total length with the fixed ratio.
- Fig. 13 sweeps `zeta_2` on the Fig. 2a setting.
- The replica batch size cap is left at Vidur's 128 unless `--batch-size-cap` is given.
- The Online LP hyperparameters default to those of `scripts/run_online_lp_policies.py`. With `reward_normalizer = 1`, every reward of at least 1 is capped at 1. `--reward-normalizer auto` divides by an upper bound of the reward instead.
