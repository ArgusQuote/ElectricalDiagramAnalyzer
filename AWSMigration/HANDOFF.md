# AWS Dev-Box Setup -- Agent Handoff

> **STATUS: PROVISIONING COMPLETE 2026-05-20 PM.** The AWS dev box is
> fully provisioned and in the "ready but not running" steady state.
> Do not re-run any of the steps below unless Marco asks for a fresh
> rebuild (e.g. a new EC2 instance). For the canonical current state,
> read `.cursor/rules/project/docs/known-issues.mdc` under "AWS dev-box
> provisioning" first -- that file is the system of record.
> The "Status (2026-05-20 PM)" and "What remains for the next agent"
> sections below tell you what's left to do, if anything.

This document is the historical execution plan from the 2026-05-20
session. Every step has a **(DONE 2026-05-20 PM)** marker followed by
what actually happened. The original recipes are kept verbatim so a
future agent can spin up another EC2 instance from scratch by
re-running the same sequence.

The upstream context (history, decisions, deferred items, and the
full operational record) is in `.cursor/rules/project/docs/known-issues.mdc`
under the "AWS dev-box provisioning" entry.

## Status (2026-05-20 PM)

- **AWS box `i-0ecb7e8fabdb8548e` (`g5.2xlarge`, `us-east-1`)** is
  fully provisioned and dormant by design.
  - SSH: `ssh -i ~/.ssh/argus-prod-key.pem ubuntu@3.89.204.0`
    (IP `3.89.204.0` is dynamic; rotates on stop/start).
  - Internal hostname: `ip-172-31-45-45`.
  - Python venv: `/home/paperspace/venv` (Python 3.10.20).
  - Repo: `/home/paperspace/ElectricalDiagramAnalyzer` on branch
    `TOOL_DEVELOPMENT_V3_MS`, HEAD `d2b545a "MS: I more mdc updates."`.
  - v7 TATR weights: `/home/paperspace/Documents/TableAnnotations/models_v7/best/`
    (111 MB, sha256 of `model.safetensors` =
    `47401faaf34ee8a797801aca5c510e1d845cf2343260b0761d9260d08c4d9ede`,
    byte-identical to Paperspace).
  - Env file: `/home/paperspace/.anvil_env` (66 bytes, mode 600,
    `paperspace:paperspace`, sha256 =
    `da78f3f9fdb43a0f1641eb7d4aedd842e7bc3118f1bd661c17d043e24a13a39b`,
    byte-identical to Paperspace).
  - `argus-uplink.service`: installed, **`disabled`**, **`inactive (dead)`**.
- **Paperspace** is on branch `TOOL_DEVELOPMENT_V3_MS` (HEAD
  `f7e53bf "Merge branch 'TOOL_Debug_MaxUserTesting' into TOOL_DEVELOPMENT_V3_MS"`
  before any local commits Marco may have made since), serving 100%
  of customer traffic. `anvil-uplink.service` is `enabled` + `active`
  and was never touched during provisioning.
- **The Anvil uplink key was not regenerated** -- the same key is
  shared between Paperspace and AWS. Running `uplink_server.py` on
  AWS will cause Anvil to round-robin jobs between the two hosts.

## What remains for the next agent

All AWS provisioning work is committed on `origin/TOOL_DEVELOPMENT_V3_MS`,
including the pip-install permission fix (`1e51cab "MS: Fixed
provision-aws-uplink.sh"`). There is no required follow-up; the items
below are optional capability tests or quality-of-life improvements.

- **(Optional)** Do a controlled foreground manual run of
  `uplink_server.py` on AWS as a capability test. This makes Anvil
  round-robin jobs between AWS and Paperspace until Ctrl-C, so it
  must NOT be run during a customer meeting or with live jobs in
  flight. Recipe is in "Step F continued -- Manual development
  run" below and in `AWSMigration/README.md` Section 6.
- **(Optional)** Stop the EC2 instance from the AWS Console to save
  money when not actively testing. `g5.2xlarge` is ~$1.21/hr
  on-demand; leaving it running 24/7 burns ~$880/mo. The provision
  script is idempotent so a stopped instance can be re-started any
  time -- only the public IP changes.
- **(Optional)** Add an `Host argus-prod-aws` block to
  `~/.ssh/config` (key `~/.ssh/argus-prod-key.pem`) so the `-i ...`
  flag isn't needed each session.

DO NOT under any circumstance:

- `systemctl enable argus-uplink` on AWS
- `systemctl start argus-uplink` on AWS
- Regenerate the `ANVIL_UPLINK_KEY` (would invalidate Paperspace's
  connection)
- Print the `ANVIL_UPLINK_KEY` value to chat or write it to the
  laptop's disk (sha256-only verification is the established pattern)
- Touch Paperspace's `anvil-uplink.service` (it's serving customers)

## Goal (achieved 2026-05-20 PM)

Provisioned the AWS EC2 instance as a **manual-run-only development /
capability-testing environment**. Paperspace remains production. AWS
does NOT auto-start the Anvil uplink under any circumstance.

Specifically:

- AWS box: fully provisioned (deps installed, repo cloned, venv built,
  weights synced, env key set, systemd unit file installed). ✓
- AWS `argus-uplink.service`: present but never `systemctl enable`d. ✓
- No EventBridge schedule. ✓
- No parallel cutover. ✓
- Paperspace's existing `anvil-uplink.service` autostart NOT touched. ✓
- Marco runs `python AnvilUplinkCode/uplink_server.py` manually on
  AWS only when actively testing. (Has not yet been done; optional.)

## Initial baseline (as of 2026-05-20 AM, pre-provisioning)

1. EC2 launched: instance ID `i-0ecb7e8fabdb8548e`, Name tag
   `argus-production-server` (the name predates the dev-only
   reclassification; do not rename), `g5.2xlarge`, `us-east-1`,
   public IP `3.89.204.0` (dynamic; will change on stop/start).
2. AMI: "Deep Learning Base GPU AMI" Marketplace listing labeled
   Ubuntu 20.04 but actual OS is **Ubuntu 24.04.3 LTS (Noble Numbat)**
   -- AWS-side packaging quirk, not a bug.
3. Hardware verified on EC2: NVIDIA A10G, 23,028 MiB VRAM, driver
   535.274.02, CUDA-driver 12.2, `nvcc` 12.0, 30 GiB RAM, 145 GiB free
   on root, no swap.
4. SSH access: `ssh -i ~/.ssh/argus-prod-key.pem ubuntu@3.89.204.0`.
   Key backed up by Marco (cloud + multiple copies).
5. Security group `argus-uplink-sg`: SSH inbound from Marco's IP only.
   Partner's IP deferred.
6. Local files created at `AWSMigration/` (now committed on
   `TOOL_DEVELOPMENT_V3_MS` per Step B):
   - `provision-aws-uplink.sh` -- idempotent bootstrap script.
   - `paperspace-freeze.txt` -- 149-package canonical install spec
     captured from production Paperspace venv.
   - `README.md` -- workflow doc.
   - `HANDOFF.md` -- this file.

## Key decisions already made (do not re-litigate)

| Decision | Choice | Rationale |
|---|---|---|
| Branch to clone on AWS | `TOOL_DEVELOPMENT_V3_MS` | Was `TOOL_Debug_MaxUserTesting` until 2026-05-20 PM; Paperspace was migrated to V3_MS (which already contained all of LW's production fixes plus the v7 ML work and the `AWSMigration/` folder) to unify Paperspace and AWS on a single branch and avoid maintaining two production trees |
| Python version | 3.10 (via deadsnakes PPA) | Match Paperspace; avoids detectron2/layoutparser 3.12 risks per the v6/v7 entries in `known-issues.mdc` |
| Install spec | `AWSMigration/paperspace-freeze.txt`, NOT `MISC/requirements.txt` | Repo `MISC/requirements.txt` is stale (pins `numpy<2` but Paperspace runs 2.1.2; transformers isn't listed) |
| `detectron2` | **Excluded** from install | Unused by `uplink_server.py` (only lazy-imported in `MLTableDetection/TableDetectorML.py:236`); fragile git-source build with CUDA-version risk |
| Code transfer to AWS | GitHub deploy key (read-only) | Cleanest long-term; future `git pull` works without ferry |
| Env file path on AWS | `/home/paperspace/.anvil_env` | Mirrors Paperspace exactly (NOT `/etc/` as the original migration plan assumed) |
| Systemd unit name | `argus-uplink.service` (AWS) vs `anvil-uplink.service` (Paperspace) | Distinguishable in logs/`journalctl` |
| EventBridge schedule | **Skip entirely** | Marco changed plan: AWS is dev, not scheduled-prod |
| `systemctl enable --now` on AWS | **Never** | Per Marco's latest direction |

## Steps remaining (in order)

### Step A -- (DONE 2026-05-20 PM)

Provisioning script footer and README rewritten to remove residual
parallel-cutover language. The script no longer ends with a
`systemctl enable` instruction; instead it has a prominent
do-not-enable warning and a manual-run command block. README
Section 5 was rewritten as "Verify ready but not running" and Section
6 as "Manual development run" (with the same warning).

### Step B -- (DONE 2026-05-20 PM)

The `AWSMigration/` folder was committed to `TOOL_DEVELOPMENT_V3_MS`
(commits `386954b`, `60cb6f7`, `c321cf5`) and pushed to origin.
Paperspace was then migrated from `TOOL_Debug_MaxUserTesting` to
`TOOL_DEVELOPMENT_V3_MS` (merge commit `f7e53bf` on origin), which
unifies Paperspace and the AWS dev box on a single production
branch. `TOOL_Debug_MaxUserTesting` is now effectively retired
(its only commits beyond the merge base were two `__pycache__/*.pyc`
binary-only bumps containing no source change, so nothing was lost
by switching).

### Step C -- (DONE 2026-05-20 PM) Run the provisioning script on AWS

**What happened**: `scp`'d the script to `/tmp/`, ran `sudo bash` on
AWS. First run exited at the deploy-key step (exit 10). Marco added
the generated ed25519 public key
(`ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIP/uHadXeMKlGyxqdBI8N8Uc2zvNupl/5OD6dwZ3idp9 argus-prod-aws-ip-172-31-45-45@20260520`)
to the repo's Deploy Keys with title `argus-prod-aws (2026-05-20)`,
write access UNCHECKED. Re-run completed Steps 4-9 (clone, venv,
pip install, weights placeholder, env placeholder, systemd unit)
in ~3 minutes. One bug encountered: the original script's pip-install
step failed with EACCES because the `mktemp` temp-freeze was
root-owned mode-600 but pip ran as `paperspace`. Fix applied
locally (added a `chown` line) and re-run succeeded.

**Recipe (for future EC2 rebuilds)**:

```bash
scp -i ~/.ssh/argus-prod-key.pem \
    AWSMigration/provision-aws-uplink.sh \
    ubuntu@<aws-host>:/tmp/

ssh -i ~/.ssh/argus-prod-key.pem ubuntu@<aws-host>
sudo bash /tmp/provision-aws-uplink.sh
```

First run will exit at Step 3 of the script (deploy-key generation)
with the public key printed. Have Marco paste it into
`https://github.com/ArgusQuote/ElectricalDiagramAnalyzer/settings/keys`:

- Title: `argus-prod-aws (<YYYY-MM-DD>)`
- Allow write access: **leave UNCHECKED** (read-only is enough)

Re-run `sudo bash /tmp/provision-aws-uplink.sh` after the deploy key
is added. The script will continue through Steps 4-9 and exit. The
pip-install permission fix is now in the committed script (or should
be -- see "What remains for the next agent" loose end #1).

### Step D -- (DONE 2026-05-20 PM) Ferry the v7 TATR weights

**What happened**: Two-step rsync ran cleanly. Paperspace -> laptop
took ~5 s at ~22 MB/s (LAN-ish via SSH); laptop -> AWS took ~4 s at
~32 MB/s. AWS-side `cp -r /tmp/v7-weights/. .../best/ && chown -R
paperspace:paperspace` succeeded. Verified by **sha256 match** on
`model.safetensors`:
`47401faaf34ee8a797801aca5c510e1d845cf2343260b0761d9260d08c4d9ede`
on both Paperspace and AWS. Local `/tmp/v7-weights-cache/` was
deleted post-transfer. One small gotcha: post-`chown`, the `ubuntu`
user can't read into `/home/paperspace/Documents` without `sudo`,
so the verification step needs `sudo du -sh` (not bare `du`).

**Recipe (for future EC2 rebuilds)**: identical to the 2026-05-20
session. From Marco's laptop:

```bash
rsync -avhP \
    paperspace-vm:/home/paperspace/Documents/TableAnnotations/models_v7/best/ \
    /tmp/v7-weights-cache/

rsync -avhP -e "ssh -i ~/.ssh/argus-prod-key.pem" \
    /tmp/v7-weights-cache/ \
    ubuntu@<aws-host>:/tmp/v7-weights/

ssh -i ~/.ssh/argus-prod-key.pem ubuntu@<aws-host> '
    sudo cp -r /tmp/v7-weights/. /home/paperspace/Documents/TableAnnotations/models_v7/best/ &&
    sudo chown -R paperspace:paperspace /home/paperspace/Documents &&
    rm -rf /tmp/v7-weights
'

rm -rf /tmp/v7-weights-cache
```

Expected size: ~111 MB total (1.4K `config.json` + 110M
`model.safetensors` + 454 `preprocessor_config.json` + 40K
`trainer_state.json` + 5.4K `training_args.bin`). Verify on AWS via
**sha256 match against Paperspace** (preferred over `du`, since size
alone doesn't catch silent corruption):

```bash
ssh -i ~/.ssh/argus-prod-key.pem ubuntu@<aws-host> \
    'sudo sha256sum /home/paperspace/Documents/TableAnnotations/models_v7/best/model.safetensors'
ssh paperspace-vm \
    'sha256sum /home/paperspace/Documents/TableAnnotations/models_v7/best/model.safetensors'
# expect identical hashes (47401faa...4d9ede as of 2026-05-20).
```

### Step E -- (DONE 2026-05-20 PM) Ferry the ANVIL_UPLINK_KEY without printing it

**What happened**: Pipe-only transfer ran cleanly. The key never
appeared in chat, never landed on the laptop's disk, never printed
to stdout (`tee` was redirected to `/dev/null`). Verified by
**sha256 match**:
`da78f3f9fdb43a0f1641eb7d4aedd842e7bc3118f1bd661c17d043e24a13a39b`
on both Paperspace and AWS (66 bytes, 1 line, starts with
`ANVIL_UPLINK_KEY=`, mode 600, `paperspace:paperspace`).

**Recipe (for future EC2 rebuilds)**:

```bash
ssh paperspace-vm 'cat /home/paperspace/.anvil_env' | \
    ssh -i ~/.ssh/argus-prod-key.pem ubuntu@<aws-host> \
        'sudo tee /home/paperspace/.anvil_env >/dev/null && \
         sudo chown paperspace:paperspace /home/paperspace/.anvil_env && \
         sudo chmod 600 /home/paperspace/.anvil_env'
```

Verify by sha256 (preferred -- proves byte-equality without revealing
the value):

```bash
ssh paperspace-vm 'sha256sum /home/paperspace/.anvil_env'
ssh -i ~/.ssh/argus-prod-key.pem ubuntu@<aws-host> \
    'sudo sha256sum /home/paperspace/.anvil_env'
# expect identical hashes (da78f3f9...43a39b as of 2026-05-20).
```

Backup verification with file metadata (no value print):

```bash
ssh -i ~/.ssh/argus-prod-key.pem ubuntu@<aws-host> \
    'sudo ls -la /home/paperspace/.anvil_env'
# expect: -rw------- 1 paperspace paperspace 66 ... .anvil_env
```

The byte count `66` matches Paperspace's file exactly.

### Step F -- (DONE 2026-05-20 PM) Verify the AWS box is "ready but not running"

**What happened**: All five smoke checks passed.

| Check | Result |
|---|---|
| Python version | `Python 3.10.20` ✓ |
| CUDA visible to PyTorch | `cuda.is_available=True`, `device=NVIDIA A10G` ✓ |
| `anvil.server` + `easyocr` + `transformers` import | clean (`transformers 4.55.4`) ✓ |
| v7 model load | OK, `num_queries=15` (correct v7 signature) ✓ |
| `argus-uplink.service` status | `disabled` + `inactive (dead)` ✓ |

Note: the v7 model load produces a stack of "for X.weight: copying
from a non-meta parameter in the checkpoint to a meta parameter in
the current model, which is a no-op" warnings from PyTorch 2.4's
safetensors loader. These are **benign noise** -- the model loaded
successfully and the same warnings appear on Paperspace + the laptop
dev RTX 500 Ada. Real load failures would raise an exception, not a
warning.

A quoting gotcha was hit on the first attempt: trying to wrap the
smoke checks in `sudo -iu paperspace bash -c "..."` over SSH with
nested escapes mangled newlines (the first line came out as
`activatecd /home/paperspace/...`). **Use a heredoc piped into
`sudo -u paperspace bash -s` instead** -- the recipe below shows
the working form.

**Recipe (for future EC2 rebuilds)**: pipe a heredoc into bash on
the remote box so quoting stays sane.

```bash
ssh -i ~/.ssh/argus-prod-key.pem ubuntu@<aws-host> 'sudo -u paperspace bash -s' <<'REMOTE'
set -u
VENV=/home/paperspace/venv
V7=/home/paperspace/Documents/TableAnnotations/models_v7/best

echo "=== Python ==="
"$VENV/bin/python" --version

echo "=== CUDA ==="
"$VENV/bin/python" -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

echo "=== imports ==="
"$VENV/bin/python" -c "import anvil.server, easyocr, transformers; print('imports ok', transformers.__version__)"

echo "=== v7 model load ==="
"$VENV/bin/python" -c "
from transformers import TableTransformerForObjectDetection
m = TableTransformerForObjectDetection.from_pretrained('$V7')
print('v7 model loads OK; num_queries=', m.config.num_queries)
"
REMOTE

# Systemd check runs as 'ubuntu' (no sudo -u needed):
ssh -i ~/.ssh/argus-prod-key.pem ubuntu@<aws-host> '
    systemctl status argus-uplink --no-pager;
    echo "is-enabled: $(systemctl is-enabled argus-uplink)";
    echo "is-active:  $(systemctl is-active argus-uplink)"
'
```

If all five checks pass, **the AWS box is in the desired state**:
ready to manually run `python AnvilUplinkCode/uplink_server.py`
whenever Marco wants to test, but does nothing at boot.

### Step F continued -- Manual development run (optional, only when Marco asks)

This is NOT a provisioning step -- it's the day-to-day use mode.
Running this makes Anvil round-robin jobs between AWS and Paperspace
until Ctrl-C, so it MUST NOT be run during a customer meeting or
with live jobs in flight.

```bash
ssh -i ~/.ssh/argus-prod-key.pem ubuntu@<aws-host>
sudo -iu paperspace
source /home/paperspace/venv/bin/activate
cd /home/paperspace/ElectricalDiagramAnalyzer
export $(cat /home/paperspace/.anvil_env | xargs)
python AnvilUplinkCode/uplink_server.py
# ... test, then Ctrl-C to stop and disconnect from Anvil.
```

Expect ~30 s startup before the 4 workers report ready. Watch for
`Connected to "Argus Automated BOM" as SERVER` to confirm Anvil
attach succeeded.

### Step G -- (DONE 2026-05-20 PM) Stopped at the right place

**What happened**: Did NOT run `python AnvilUplinkCode/uplink_server.py`.
Did NOT run `systemctl enable --now argus-uplink`. The box was left
in the "ready but not running" steady state and the session ended.
Marco was asked whether he wanted to do an immediate manual
foreground test or leave the box dormant; the answer (implicitly,
by not running it) was "leave it dormant for now."

**Guardrails for the next agent (carry forward, non-negotiable)**:

- Do NOT run `python AnvilUplinkCode/uplink_server.py` on AWS unless
  Marco explicitly requests it -- running it makes Anvil round-robin
  between AWS and Paperspace for as long as the process is up.
- Do NOT run `systemctl enable --now argus-uplink`. The systemd unit
  is present for symmetry but must never be activated under the
  current plan.

## Critical guardrails (carry forward, non-negotiable)

1. **Paperspace's `anvil-uplink.service` is untouchable.** Marco
   explicitly does not want it disabled, stopped, or modified.
   Read-only access to its config and env file is fine.
2. **Never run `systemctl enable --now argus-uplink` on AWS.** Even
   after Step F passes. Marco must explicitly request it as a
   separate decision later, and the current plan is for him never
   to make that request.
3. **Never regenerate the `ANVIL_UPLINK_KEY`** in the Anvil
   dashboard. Doing so invalidates Paperspace's connection. Copying
   the existing key (Step E) is the only correct path.
4. **Never print the `ANVIL_UPLINK_KEY` to chat or stdout.** Use the
   pipe-only transfer in Step E.
5. **Customer meetings are happening.** Do not run
   `python AnvilUplinkCode/uplink_server.py` on AWS during business
   hours unless Marco explicitly says so -- running it makes Anvil
   round-robin between AWS and Paperspace.
6. **Do not commit anything without Marco's confirmation.** Per the
   workspace's `git-conventions` rule and Cursor's default policy.

## Open items for Marco's future decisions (not blockers)

These should land as deferred items in `known-issues.mdc`'s
"AWS dev-box provisioning" entry, not as automatic next actions:

- Add Marco's business partner's IP to the `argus-uplink-sg` security
  group when the partner's IP is known.
- Decide if/when AWS should be promoted to production. (Currently no
  plan.)
- Decide if/when Paperspace's autostart should be disabled.
  (Currently keep it on.)
- Decide if/when EventBridge scheduling is wanted. (Currently not
  wanted.)
- Decide if Elastic IP is worth setting up. (Currently using dynamic
  public IP; changes on stop/start.)
- Decide if a separate Anvil app for staging is worth setting up.
  (Currently AWS and Paperspace share the same uplink key.)
- Fix the long-standing `MISC/requirements.txt` drift vs production.
  (Not part of this migration; tracked separately.)

## What Marco does day-to-day after Step F passes

```bash
# To test changes on AWS dev:
ssh -i ~/.ssh/argus-prod-key.pem ubuntu@<current-public-ip>
sudo -iu paperspace
source /home/paperspace/venv/bin/activate
cd /home/paperspace/ElectricalDiagramAnalyzer
export $(cat /home/paperspace/.anvil_env | xargs)
python AnvilUplinkCode/uplink_server.py
# ... test, Ctrl-C when done
```

The current public IP must be looked up in the AWS Console each
session (it's dynamic). If Marco stops the EC2 instance to save
money, the next session will require:

1. Start the instance from the AWS Console.
2. Wait ~2 min for `2/2 status checks` to pass.
3. Copy the new Public IPv4 from the console.
4. SSH using that new IP.

## Useful host aliases on Marco's laptop

The laptop already has these SSH config entries:

```
Host paperspace-vm
    HostName 184.105.3.207
    User paperspace
    # (uses ~/.ssh/id_ed25519)
```

(No alias yet for AWS -- always pass `-i ~/.ssh/argus-prod-key.pem
ubuntu@<ip>` explicitly. Adding an `Host argus-prod-aws` alias is a
nice future quality-of-life improvement, but optional.)
