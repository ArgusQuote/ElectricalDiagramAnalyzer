# AWS Dev-Box (NOT a production migration)

Scripts and reference material for the **AWS dev / capability-testing
environment** (`g5.2xlarge`, `us-east-1`, Ubuntu 24.04). Paperspace
remains the sole customer-facing production uplink. AWS is a
manual-run-only sandbox: the systemd unit is installed but
**deliberately never enabled**.

Full context (decisions, history, deferred items) lives in
`.cursor/rules/project/docs/known-issues.mdc` under the
"AWS dev-box provisioning" entry. Step-by-step execution plan
for the next agent is in `HANDOFF.md`.

## Files in this directory

| File | Purpose |
|---|---|
| `provision-aws-uplink.sh` | Idempotent bootstrap script run on the AWS box as `root`. Installs deps, creates `paperspace` user, generates GitHub deploy key, clones the repo, creates the venv, writes the systemd unit. Does NOT start, enable, or otherwise auto-run the uplink. |
| `paperspace-freeze.txt` | Canonical `pip freeze` from production Paperspace. Used as the authoritative install spec for the AWS venv to avoid drift between `MISC/requirements.txt` and actual production. |
| `HANDOFF.md` | Concrete step-by-step execution plan, designed to be the first file a new agent reads when resuming this work. |

## End-to-end workflow

This summary assumes the EC2 instance is already launched (Step 1 of the
migration plan) and you can SSH in as `ubuntu`.

1. **Copy the script up.** From your laptop:
   ```bash
   scp -i ~/.ssh/argus-prod-key.pem \
       AWSMigration/provision-aws-uplink.sh \
       ubuntu@<aws-host>:/tmp/
   ```

2. **Run the script.** On the EC2 box:
   ```bash
   ssh -i ~/.ssh/argus-prod-key.pem ubuntu@<aws-host>
   sudo bash /tmp/provision-aws-uplink.sh
   ```

   On the first run, it stops at Step 3 with a public key it wants you to
   paste into the repo's Deploy Keys settings:
   `https://github.com/ArgusQuote/ElectricalDiagramAnalyzer/settings/keys`

   After adding the key in GitHub, re-run the same command. It picks up
   where it left off, clones the repo, builds the venv, and so on.

3. **Sync the v7 TATR weights.** The provisioning script does NOT pull
   the v7 weights -- they live at `~/Documents/TableAnnotations/models_v7/`
   on Paperspace and need to be ferried over. From your laptop (two-step
   via the laptop as a bridge):
   ```bash
   rsync -avhP \
       paperspace-vm:/home/paperspace/Documents/TableAnnotations/models_v7/best/ \
       /tmp/v7-weights-cache/
   rsync -avhP -e "ssh -i ~/.ssh/argus-prod-key.pem" \
       /tmp/v7-weights-cache/ \
       ubuntu@<aws-host>:/tmp/v7-weights/
   ```
   Then on AWS:
   ```bash
   sudo mv /tmp/v7-weights/* \
       /home/paperspace/Documents/TableAnnotations/models_v7/best/
   sudo chown -R paperspace:paperspace /home/paperspace/Documents
   rm -rf /tmp/v7-weights /tmp/v7-weights-cache
   ```

4. **Set the ANVIL_UPLINK_KEY.** Retrieve the key from Paperspace's
   `/home/paperspace/.anvil_env` (the value, not the variable name), then
   on AWS:
   ```bash
   sudo nano /home/paperspace/.anvil_env
   # Replace PASTE_KEY_HERE with the literal key. Save (Ctrl-O, Ctrl-X).
   ```
   Do NOT regenerate the key in Anvil's UI -- that would invalidate the
   Paperspace uplink. Both AWS and Paperspace use the same key by design;
   Anvil round-robins jobs between them.

5. **Verify "ready but not running".** Confirms imports and GPU access
   without starting the uplink. Safe to run at any time, including
   during business hours.
   ```bash
   sudo -iu paperspace
   source /home/paperspace/venv/bin/activate
   cd /home/paperspace/ElectricalDiagramAnalyzer
   python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   # expect: True NVIDIA A10G
   python -c "import anvil.server, easyocr, transformers; print('imports ok')"
   python -c "
   from transformers import TableTransformerForObjectDetection
   m = TableTransformerForObjectDetection.from_pretrained('/home/paperspace/Documents/TableAnnotations/models_v7/best')
   print('v7 model loads OK, num_queries=', m.config.num_queries)"
   # expect: v7 model loads OK, num_queries= 15

   systemctl status argus-uplink --no-pager
   # expect: Loaded ... disabled    Active: inactive (dead)
   ```

6. **Manual development run** (ONLY when Marco wants to actively test).
   Running this WILL connect to Anvil, which will then route ~half
   of new customer jobs to AWS until Ctrl-C. Do not run during
   customer meetings or peak business hours.
   ```bash
   sudo -iu paperspace
   source /home/paperspace/venv/bin/activate
   cd /home/paperspace/ElectricalDiagramAnalyzer
   export $(cat /home/paperspace/.anvil_env | xargs)
   python AnvilUplinkCode/uplink_server.py
   # Watch for: 'Connected to "Argus Automated BOM" as SERVER'
   # ... test, then Ctrl-C to stop and disconnect.
   ```

   **Do NOT run `sudo systemctl enable --now argus-uplink`.** The
   systemd unit is present for symmetry with Paperspace's setup, but
   AWS is explicitly configured to NOT auto-start the uplink. Running
   it via systemd would auto-restart on every reboot, which is the
   opposite of the current "manual-run only" goal.

## Paperspace safety guarantees

Every step in this folder is designed so Paperspace's
`anvil-uplink.service` is not affected.

- The provisioning script touches only the AWS box.
- The rsync of v7 weights is a read-only pull from Paperspace.
- The `ANVIL_UPLINK_KEY` is copied, never rotated.
- The AWS systemd service is installed but **never enabled or started
  automatically**; manual runs are explicit and ephemeral.
- During a manual run, Anvil round-robins jobs across the two
  connected uplinks. If AWS misbehaves mid-run, Ctrl-C in the manual
  session returns 100% of traffic to Paperspace within seconds.

## What gets excluded from the freeze on AWS

The provisioning script installs all 149 lines of `paperspace-freeze.txt`
EXCEPT one:

- **`detectron2`** -- installed from a git source build on Paperspace.
  Production does NOT use it; the default `TableDetectorML` backend is
  `table-transformer` (per `MLTableDetection/TableDetectorML.py:94`).
  Skipping it on AWS avoids a 20-30 min compile and a real risk of
  failure due to nvcc-version mismatch (Paperspace has nvcc 12.1, AWS
  Deep Learning Base GPU AMI has nvcc 12.0). If the ML pipeline ever
  needs detectron2 on AWS, install it manually per the
  [detectron2 docs](https://detectron2.readthedocs.io/en/latest/tutorials/install.html).

## Updating the canonical install spec

If you upgrade a package on Paperspace and want AWS to inherit the change,
regenerate the freeze file:

```bash
ssh paperspace-vm '/home/paperspace/venv/bin/pip freeze' > paperspace-freeze.txt
git add AWSMigration/paperspace-freeze.txt
git commit -m "Refresh Paperspace pip freeze for AWS install spec"
```

Then on AWS, pull and reinstall:

```bash
ssh ubuntu@<aws-host>
sudo -iu paperspace
cd ElectricalDiagramAnalyzer && git pull
rm /home/paperspace/venv/.argus-install-complete
sudo bash /tmp/provision-aws-uplink.sh   # script will reinstall
```
