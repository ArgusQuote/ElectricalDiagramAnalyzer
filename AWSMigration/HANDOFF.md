# AWS Dev-Box Setup -- Agent Handoff

This document is the entry point for the next agent picking up where
the 2026-05-20 session left off. Read it before doing anything else.

The upstream context (history, decisions, deferred items) is in
`.cursor/rules/project/docs/known-issues.mdc` under the "AWS dev-box
provisioning" entry; this file is the **operational plan**, that one
is the **system of record**.

## Goal (revised 2026-05-20 PM)

Provision the AWS EC2 instance as a **manual-run-only development /
capability-testing environment**. Paperspace remains production. AWS
does NOT auto-start the Anvil uplink under any circumstance.

Specifically:

- AWS box: fully provisioned (deps installed, repo cloned, venv built,
  weights synced, env key set, systemd unit file installed).
- AWS `argus-uplink.service`: present but **never** `systemctl enable`d.
- No EventBridge schedule.
- No parallel cutover.
- Paperspace's existing `anvil-uplink.service` autostart is **NOT**
  touched (decision deferred).
- Marco runs `python AnvilUplinkCode/uplink_server.py` manually on
  AWS only when actively testing.

## What's already done (as of 2026-05-20 PM)

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
6. Local files created at `AWSMigration/` (NOT committed to git yet):
   - `provision-aws-uplink.sh` -- idempotent bootstrap script.
   - `paperspace-freeze.txt` -- 149-package canonical install spec
     captured from production Paperspace venv.
   - `README.md` -- workflow doc.
   - `HANDOFF.md` -- this file.

## Key decisions already made (do not re-litigate)

| Decision | Choice | Rationale |
|---|---|---|
| Branch to clone on AWS | `TOOL_Debug_MaxUserTesting` | Marco confirmed this is the production branch on Paperspace |
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

### Step B -- Commit the AWSMigration/ folder

The provisioning script must be on disk on AWS to run, and the
cleanest way is via `git clone`. Currently the files exist only on
Marco's laptop. Commit them on the same branch Paperspace runs
(`TOOL_Debug_MaxUserTesting`):

```bash
git add AWSMigration/
git commit -m "Add AWS dev-box provisioning script and pinned install spec"
git push origin TOOL_Debug_MaxUserTesting
```

**Get Marco's confirmation before pushing** -- the workspace's
git-conventions rule forbids commits without explicit user consent.

### Step C -- Run the provisioning script on AWS

```bash
scp -i ~/.ssh/argus-prod-key.pem \
    AWSMigration/provision-aws-uplink.sh \
    ubuntu@3.89.204.0:/tmp/

ssh -i ~/.ssh/argus-prod-key.pem ubuntu@3.89.204.0
sudo bash /tmp/provision-aws-uplink.sh
```

First run will exit at Step 3 of the script (deploy-key generation)
with the public key printed. Have Marco paste it into
`https://github.com/ArgusQuote/ElectricalDiagramAnalyzer/settings/keys`:

- Title: `argus-prod-aws-dev`
- Allow write access: **leave UNCHECKED** (read-only is enough)

Re-run `sudo bash /tmp/provision-aws-uplink.sh` after the deploy key
is added. The script will continue through Steps 4-9 and exit.

### Step D -- Ferry the v7 TATR weights

The next agent has SSH access to both `paperspace-vm` (alias in
`~/.ssh/config`, key `~/.ssh/id_ed25519`) and the AWS box (key
`~/.ssh/argus-prod-key.pem`). Run from Marco's laptop:

```bash
rsync -avhP \
    paperspace-vm:/home/paperspace/Documents/TableAnnotations/models_v7/best/ \
    /tmp/v7-weights-cache/

rsync -avhP -e "ssh -i ~/.ssh/argus-prod-key.pem" \
    /tmp/v7-weights-cache/ \
    ubuntu@3.89.204.0:/tmp/v7-weights/

ssh -i ~/.ssh/argus-prod-key.pem ubuntu@3.89.204.0 '
    sudo mkdir -p /home/paperspace/Documents/TableAnnotations/models_v7/best &&
    sudo cp -r /tmp/v7-weights/* /home/paperspace/Documents/TableAnnotations/models_v7/best/ &&
    sudo chown -R paperspace:paperspace /home/paperspace/Documents &&
    rm -rf /tmp/v7-weights
'

rm -rf /tmp/v7-weights-cache
```

Expected size: ~111 MB total (1.4K `config.json` + 110M
`model.safetensors` + 454 `preprocessor_config.json` + 40K
`trainer_state.json` + 5.4K `training_args.bin`). Verify on AWS:

```bash
ssh -i ~/.ssh/argus-prod-key.pem ubuntu@3.89.204.0 \
    'du -sh /home/paperspace/Documents/TableAnnotations/models_v7/best/'
# expect: 111M  /home/paperspace/Documents/TableAnnotations/models_v7/best/
```

### Step E -- Ferry the ANVIL_UPLINK_KEY without printing it

Pipe-only transfer; never lands on disk on the laptop, never appears
in chat:

```bash
ssh paperspace-vm 'cat /home/paperspace/.anvil_env' | \
    ssh -i ~/.ssh/argus-prod-key.pem ubuntu@3.89.204.0 \
        'sudo tee /home/paperspace/.anvil_env >/dev/null && \
         sudo chown paperspace:paperspace /home/paperspace/.anvil_env && \
         sudo chmod 600 /home/paperspace/.anvil_env'
```

Verify with file metadata (no value print):

```bash
ssh -i ~/.ssh/argus-prod-key.pem ubuntu@3.89.204.0 \
    'ls -la /home/paperspace/.anvil_env'
# expect: -rw------- 1 paperspace paperspace 66 ... .anvil_env
```

The byte count `66` matches Paperspace's file exactly.

### Step F -- Verify the AWS box is "ready but not running"

```bash
ssh -i ~/.ssh/argus-prod-key.pem ubuntu@3.89.204.0
sudo -iu paperspace
source /home/paperspace/venv/bin/activate
cd /home/paperspace/ElectricalDiagramAnalyzer

# Smoke tests (do NOT start uplink_server.py)
python --version
# expect: 3.10.x

python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# expect: True NVIDIA A10G

python -c "import anvil.server, easyocr, transformers; print('imports ok')"
# expect: imports ok

python -c "
from transformers import TableTransformerForObjectDetection
m = TableTransformerForObjectDetection.from_pretrained('/home/paperspace/Documents/TableAnnotations/models_v7/best')
print('v7 model loads OK, num_queries=', m.config.num_queries)"
# expect: v7 model loads OK, num_queries= 15

# Confirm systemd unit is disabled (the key safety property)
systemctl status argus-uplink --no-pager
# expect: Loaded: ... disabled
#         Active: inactive (dead)
```

If all five checks pass, **the AWS box is in the desired state**:
ready to manually run `python AnvilUplinkCode/uplink_server.py`
whenever Marco wants to test, but does nothing at boot.

### Step G -- STOP HERE

Do NOT run `python AnvilUplinkCode/uplink_server.py` unless Marco
explicitly requests it -- running it makes Anvil round-robin between
AWS and Paperspace for as long as the process is up.

Do NOT run `systemctl enable --now argus-uplink`. The systemd unit
is present for symmetry but must never be activated under the current
plan.

Confirm with Marco that the handoff is complete and ask whether he
wants to do an immediate test session (a controlled foreground run
of `uplink_server.py`) or leave the box dormant for now.

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
