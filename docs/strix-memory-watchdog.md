# Strix host-memory watchdog

`scripts/strix_memory_watchdog.py` is an external Linux command wrapper for headless Strix Halo validation. It does not change model loading or cache sizing. It measures host-wide memory from procfs and controls the launched command's process group.

```sh
./scripts/strix_memory_watchdog.py -- ./build/bin/llama-server <arguments>
```

The wrapper performs these checks and actions:

- It refuses to launch if `/proc/swaps` contains any active entry.
- It calculates used memory as `MemTotal - MemAvailable`. Linux reports these fields in KiB, so the wrapper multiplies each value by 1024 and keeps all accounting as integer bytes.
- It sends `SIGTERM` to the process group at 116 GiB used.
- It sends `SIGKILL` at 118 GiB used or 30 seconds after `SIGTERM`.
- It reports `grace_timeout` if any descendant requires `SIGKILL` after the soft-threshold grace period, even when the direct child exited earlier.
- It sends `SIGKILL` and fails if swap appears or required procfs data becomes unavailable during execution.
- It forwards wrapper `SIGHUP`, `SIGINT`, or `SIGTERM` to the process group, waits the configured grace period, then sends `SIGKILL` if any group member remains.
- It checks the process group after the direct child exits and cleans up remaining descendants before returning the child's classification.
- It applies the same bounded process-group cleanup if an unexpected post-launch error occurs.
- It propagates an unmonitored child exit code. A signal exit uses the shell convention `128 + signal`.

The 118 GiB emergency threshold leaves a 2 GiB sampling margin below the strict 120 GiB ceiling. The default sample interval is one second. This margin cannot guarantee the ceiling for a workload that can allocate more than 2 GiB between samples. Lower `--emergency-gib` or shorten `--sample-interval-seconds` for such a workload.

Use `--procfs-root` to select a different procfs mount or a test fixture. `--soft-gib`, `--emergency-gib`, `--grace-seconds`, and `--sample-interval-seconds` override the other defaults. The emergency threshold must remain below 120 GiB.

The wrapper writes timestamped JSON Lines records to standard error. Preflight, sample, signal, and final records include total, available, used, and peak-used bytes, swap entry count, child status, process-group status, threshold reason, and final classification where applicable. Signal records are written immediately after each process-group signal. Child standard input, standard output, and standard error are inherited unchanged.

## Watchdog-owned validation lease

Use all three artifact options together when another process must prove that it is inside the active watchdog process group:

```sh
./scripts/strix_memory_watchdog.py \
    --lease-path /run/deepseek-v41/watchdog-lease.json \
    --heartbeat-path /run/deepseek-v41/watchdog-heartbeat.json \
    --audit-path /run/deepseek-v41/watchdog-audit.jsonl \
    -- \
    python3 tools/deepseek-v41-trace/run_matrix.py <arguments>
```

The watchdog creates the persistent audit before launch, then atomically creates the lease and heartbeat after `Popen` returns. Existing artifact paths are rejected rather than overwritten. The child receives the resolved paths through `STRIX_MEMORY_WATCHDOG_LEASE_PATH`, `STRIX_MEMORY_WATCHDOG_HEARTBEAT_PATH`, and `STRIX_MEMORY_WATCHDOG_AUDIT_PATH`. It also receives `STRIX_MEMORY_WATCHDOG_HEARTBEAT_MAX_AGE_SECONDS`.

The child can run before the first atomic lease rename. A matching preflight must retry the inherited lease path for a bounded interval and fail closed if a complete valid lease does not appear. It must not accept a lease path supplied separately by the operator.

Lease format `strix-memory-watchdog-lease`, version 1, contains:

- `lease_id` and active/final `state`
- `watchdog_pid`, `watchdog_start_time_utc`, Linux `watchdog_start_time_ticks`, `watchdog_command_sha256`, `watchdog_script_path`, and `watchdog_script_sha256`
- exact `soft_bytes`, `emergency_bytes`, and `strict_ceiling_bytes`
- `procfs_root`
- `child_pid`, `child_process_group_id`, `command`, and `child_command_sha256`
- `heartbeat_path`, `max_heartbeat_age_seconds`, and `audit_path`
- the authoritative `final` audit record after termination

Heartbeat format `strix-memory-watchdog-heartbeat`, version 1, binds `lease_id`, watchdog PID/start ticks, child PID/process group, sequence, state, and update timestamps. Every memory sample atomically replaces the heartbeat and includes the complete sample audit record. A final heartbeat and final lease update remain on disk with the persistent JSONL audit; the watchdog does not delete this evidence.

A matching Linux preflight must verify all of the following:

- The inherited lease, heartbeat, and audit paths match the paths inside the lease.
- The expected repository script path hashes to `watchdog_script_sha256`.
- `/proc/<watchdog_pid>/stat` start ticks and `/proc/<watchdog_pid>/cmdline` SHA-256 match the lease.
- The watchdog command line names the expected script.
- The current process group equals `child_process_group_id`, whose leader is `child_pid` and whose parent is `watchdog_pid`.
- The command identity is expected, the procfs root is `/proc`, and thresholds are exactly 116 GiB soft, 118 GiB emergency, and 120 GiB strict ceiling for the final run.
- The heartbeat identity matches the lease and its monotonic timestamp is not older than `max_heartbeat_age_seconds`.
- The persistent audit exists and contains watchdog JSONL records.

These checks bind the validation process to the live canonical watchdog. A standalone heartbeat helper has a different PID, start time, command line, script hash, and process group and cannot satisfy the lease.

Exit classifications are authoritative in the final JSON record. Operational failures use these exit codes:

| Exit code | Classification |
| ---: | --- |
| 2 | procfs or configuration error |
| 3 | swap active at startup or detected during execution |
| 4 | soft threshold reached |
| 5 | emergency threshold reached |
| 6 | soft-threshold grace period expired |
| 7 | process-group signaling or termination failure |
| 8 | lease, heartbeat, or persistent audit failure |
| 70 | unexpected post-launch error |
| 127 | command launch failure |

No model, backend, or ROCm package is required to run the unit tests:

```sh
python3 tests/test_strix_memory_watchdog.py
```
