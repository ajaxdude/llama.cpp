# Strix Halo host restoration fail-safe

This package is an inert, model-load-free restoration fail-safe for the known Fedora x86_64 AMD Ryzen AI Max Strix Halo host. It does not capture a baseline, install systemd units, arm a timer, access model files, stop a service, disable a service, or perform network operations.

Installation and arming require separate, explicit administrator action and authentication. Possession of these files, a generated plan, a validation result, or this documentation is not authorization to install or execute the restore service.

## Safety model

`strix-host-restore` accepts only an exact HMAC-SHA256-authenticated JSON plan. Restore mode requires root and fails closed unless all of these conditions hold:

- The host reports Fedora, x86_64, and an `AMD Ryzen AI Max` CPU.
- The plan, signature, and 32-byte hexadecimal HMAC key are regular files owned by root, mode `0600`, with one hard link and no symlink in any path component.
- The plan has the complete schema below, no duplicate or unknown keys, the fixed platform value, the armed nonce, and the armed deadline.
- The deadline has arrived, is no more than seven days after validation, and the plan expires no more than one hour after the deadline.
- The audit directory is a real root-owned `0700` directory. Existing audit and lock files must be safe regular files.
- The target user exists. `/run/user/<uid>` must be a UID-owned `0700` directory and its `bus` must be a UID-owned Unix socket.
- Every observed Docker and systemd state is recognized. A missing Docker health check, malformed output, command failure, timeout, or audit failure stops later restoration steps.

The script only moves state toward baseline `enabled` and `active` values. It contains no stop, disable, swapoff, remove, restart, or container recreation path.

Restoration order is fixed:

1. Enable `/swapfile` with `swapon` if it was enabled in the baseline.
2. Start the existing Docker container `monerod` if it was running in the baseline, then wait for its existing Docker health check to report `healthy`.
3. Reconcile `p2pool.service`.
4. Reconcile `xmrig.service`.
5. Reconcile the target user's `llama-swap.service`.
6. Reconcile the target user's `llama-proxy.service`.

Each component checkpoint is written to `/var/lib/strix-host-restore/restore-<nonce>.json` with atomic replacement, `fsync`, no-follow opens, and mode `0600`. Re-running the same plan is safe after a partial restoration. Already-restored state is recorded without repeating the mutation.

## Plan format

All keys shown are mandatory. Component names, unit names, the Docker container, and `/swapfile` are intentionally not configurable.

```json
{
  "schema": 1,
  "platform": {
    "os": "fedora",
    "architecture": "x86_64",
    "device": "strix-halo"
  },
  "nonce": "replace_with_16_to_64_safe_characters",
  "restore_deadline_epoch": 2000000060,
  "expires_epoch": 2000000660,
  "target_uid": 1000,
  "components": {
    "swapfile": {
      "enabled": true
    },
    "docker_monerod": {
      "running": true,
      "health_timeout_seconds": 120
    },
    "p2pool": {
      "active": true,
      "enabled": true
    },
    "xmrig": {
      "active": true,
      "enabled": true
    },
    "llama_swap": {
      "active": true,
      "enabled": true
    },
    "proxy": {
      "active": true,
      "enabled": true
    }
  }
}
```

The coordinator must capture these booleans before it changes host state. `active: false`, `enabled: false`, `running: false`, and `swapfile.enabled: false` are explicit no-op instructions. They never authorize the restore script to stop or disable an already-running component.

The `target_uid` is mandatory even when both user services are inactive and disabled. The Docker health timeout must be from 1 through 600 seconds. The plan is authenticated over its exact bytes, so formatting changes after signing invalidate it.

## Harmless non-root validation

Validation executes no host commands and performs no mutations. It expects a user-owned `0600` plan, signature, and key so the same ownership checks can run without root. Use a copy of the exact staged plan and signature, not the inaccessible installed root files.

```sh
chmod 0600 ./plan.json ./plan.sig ./plan.key
./strix-host-restore validate \
    --plan ./plan.json \
    --signature ./plan.sig \
    --key ./plan.key \
    --expected-nonce restore_nonce_123456 \
    --expected-deadline 2000000060
```

Successful output starts with `VALID`, prints the plan SHA-256 and ordered intended actions, and ends with `No commands were executed.` Validation permits a future deadline, but still rejects an expired plan or a deadline more than seven days away.

For a 32-byte hexadecimal key in `plan.key`, the detached signature is lowercase hexadecimal HMAC-SHA256:

```sh
python3 -c 'import hashlib,hmac,pathlib,sys; key=bytes.fromhex(pathlib.Path(sys.argv[1]).read_text().strip()); data=pathlib.Path(sys.argv[2]).read_bytes(); print(hmac.new(key,data,hashlib.sha256).hexdigest())' plan.key plan.json > plan.sig
chmod 0600 plan.sig
```

The command only reads the key and plan and writes the requested signature. Generate and protect the key through the administrator's normal secret-handling process. Do not reuse it outside this fail-safe.

## Administrator installation and arming

The files ending in `.in` are templates, not installable live units. A human administrator must review the captured plan and render all placeholders before copying anything:

- Replace `@EXPECTED_NONCE@` and `@EXPECTED_DEADLINE_EPOCH@` in the service with the exact signed plan values.
- Replace `@RESTORE_DEADLINE_UTC@` in the timer with the same instant in a systemd `OnCalendar` UTC form, for example `2033-05-18 03:34:20 UTC`.
- Install the script as `/usr/local/sbin/strix-host-restore`, owned by root and mode `0755`.
- Install the reviewed plan, signature, and key under `/etc/strix-host-restore`, owned by root and mode `0600`.
- Create `/var/lib/strix-host-restore` as a real root-owned directory with mode `0700`.
- Install the rendered service and timer under `/etc/systemd/system`, then explicitly enable and start only the timer. The service template has no `[Install]` target and cannot be enabled on its own.

The timer uses `Persistent=true`, one-second accuracy, and no random delay. It invokes the service at the rendered deadline even if the coordinator, SSH connection, or workload session has ended. The service does not pull in or start `docker.service`; it only orders itself after Docker if Docker is already part of the active host baseline. The service independently checks the signed epoch and nonce, so a timer rendered for another plan fails closed. A failed run remains visible in systemd and the durable audit record; an administrator can re-run the idempotent service after correcting a transient failure, but the signed expiry is checked throughout restoration.

Do not arm the timer until the non-root validation output, rendered unit values, baseline capture, target UID, Docker health check, audit path, and deadline have been reviewed. Installing the service alone does not arm it.

## Tests

The focused suite uses only Python's standard library and fake command execution:

```sh
python3 -m unittest -v scripts/strix-host-restore/test_strix_host_restore.py
```

It covers every component, required ordering, repeat runs, inactive baselines, strict/malformed/tampered plans, ownership and link checks, plan and audit symlink attacks, command failures, Docker health failures and timeout, explicit user UID transport, unsupported states, and audit write failures.
