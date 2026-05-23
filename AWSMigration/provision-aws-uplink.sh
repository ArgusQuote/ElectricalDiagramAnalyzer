#!/usr/bin/env bash
# AWSMigration/provision-aws-uplink.sh
#
# Provisions a fresh EC2 instance as the AWS dev / capability-testing
# environment for the "Argus Automated BOM" Anvil uplink server.
#
# Target host:
#   - Ubuntu 24.04 (Noble Numbat) on g5.2xlarge with the "Deep Learning Base
#     GPU AMI (Ubuntu 20.04)" Marketplace image (the OS is actually 24.04
#     despite the AMI listing's name; verified 2026-05-20).
#   - Runs as the default 'ubuntu' user via sudo. No additional user is
#     created. (Until 2026-05-23 this script created a separate 'paperspace'
#     user to mirror the Paperspace VM exactly; that turned out to be
#     unnecessary indirection and was retired on 2026-05-23 -- see
#     known-issues.mdc "AWS dev-box provisioning" entry, subsection
#     "Operational notes from the 2026-05-23 session".)
#
# Usage (on the EC2 box, as the ubuntu user):
#   sudo bash /tmp/provision-aws-uplink.sh
#
# This script is IDEMPOTENT. Re-running it is safe and picks up wherever the
# previous run stopped. If the GitHub deploy key has not yet been added, the
# script prints the public key and exits with code 10; re-run after adding the
# key to https://github.com/ArgusQuote/ElectricalDiagramAnalyzer/settings/keys
#
# This script DELIBERATELY does NOT:
#   - systemctl enable --now argus-uplink   (you do this manually after smoke test)
#   - rsync the v7 TATR weights from Paperspace (laptop-mediated, separate step)
#   - set ANVIL_UPLINK_KEY in /home/ubuntu/.anvil_env (manual paste after
#     retrieval from Paperspace; the key is a long-lived secret and must never
#     appear in a script or chat log)
#
# Paperspace-safety: every step in this script touches the AWS box only.
# The Paperspace VM is not contacted at any point. Paperspace's
# anvil-uplink.service stays running, untouched, throughout.

set -euo pipefail

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
# Service account that owns the venv, repo, env file, and (when run)
# the uplink process. The AMI's default 'ubuntu' user is used directly --
# it already exists, has NOPASSWD sudo, and has the SSH key Marco uses
# to reach the box, so no additional user-creation step is needed.
UPLINK_USER="ubuntu"
UPLINK_HOME="/home/${UPLINK_USER}"
REPO_URL="git@github.com:ArgusQuote/ElectricalDiagramAnalyzer.git"
REPO_BRANCH="TOOL_DEVELOPMENT_V3_MS"
REPO_DIR="${UPLINK_HOME}/ElectricalDiagramAnalyzer"
VENV_DIR="${UPLINK_HOME}/venv"
ENV_FILE="${UPLINK_HOME}/.anvil_env"
SYSTEMD_UNIT="/etc/systemd/system/argus-uplink.service"
PYTHON_VERSION="3.10"
FREEZE_FILE="${REPO_DIR}/AWSMigration/paperspace-freeze.txt"

# -----------------------------------------------------------------------------
# Logging helpers
# -----------------------------------------------------------------------------
log()  { printf '\n\033[1;34m[%s]\033[0m %s\n' "$(date +%H:%M:%S)" "$*"; }
ok()   { printf '  \033[1;32mOK\033[0m   %s\n' "$*"; }
skip() { printf '  \033[1;33mSKIP\033[0m %s (already done)\n' "$*"; }
warn() { printf '  \033[1;33mWARN\033[0m %s\n' "$*"; }
die()  { printf '\n\033[1;31mERROR\033[0m %s\n' "$*" >&2; exit 1; }

# -----------------------------------------------------------------------------
# Pre-checks
# -----------------------------------------------------------------------------
[ "$(id -u)" = "0" ] || die "Run as root: sudo bash $0"
[ -f /etc/os-release ] && grep -q 'Ubuntu' /etc/os-release \
  || die "This script targets Ubuntu. Detected: $(cat /etc/os-release | head -1)"
command -v nvidia-smi >/dev/null \
  || die "nvidia-smi not found. Wrong AMI?"
id "$UPLINK_USER" >/dev/null 2>&1 \
  || die "Expected user '${UPLINK_USER}' to already exist on this AMI."

log "Pre-checks OK on $(hostname). GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

# -----------------------------------------------------------------------------
# Step 1: APT packages (build tools, git, deadsnakes for python3.10)
# -----------------------------------------------------------------------------
log "Step 1/8: APT packages"

if ! command -v python3.10 >/dev/null; then
  apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
    git rsync curl wget \
    build-essential \
    ghostscript poppler-utils \
    software-properties-common \
    ca-certificates
  ok "base packages installed"

  # deadsnakes PPA for Python 3.10 on Noble (24.04 ships 3.12 by default)
  add-apt-repository -y ppa:deadsnakes/ppa >/dev/null
  apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
    python3.10 python3.10-venv python3.10-dev
  ok "python3.10 installed: $(python3.10 --version)"
else
  skip "python3.10 already installed: $(python3.10 --version)"
fi

# -----------------------------------------------------------------------------
# Step 2: GitHub deploy key (generate, prompt user to add, verify)
# -----------------------------------------------------------------------------
log "Step 2/8: GitHub deploy key for repo clone"

DEPLOY_KEY="${UPLINK_HOME}/.ssh/id_ed25519"
if [ ! -f "$DEPLOY_KEY" ]; then
  sudo -u "$UPLINK_USER" mkdir -p "${UPLINK_HOME}/.ssh"
  sudo -u "$UPLINK_USER" chmod 700 "${UPLINK_HOME}/.ssh"
  sudo -u "$UPLINK_USER" ssh-keygen -t ed25519 \
    -C "argus-prod-aws-${UPLINK_USER}-$(hostname)@$(date +%Y%m%d)" \
    -f "$DEPLOY_KEY" -N "" >/dev/null
  ok "deploy key generated"
fi

# Trust github.com host key (idempotent)
KNOWN_HOSTS="${UPLINK_HOME}/.ssh/known_hosts"
if [ ! -f "$KNOWN_HOSTS" ] || ! grep -q 'github.com' "$KNOWN_HOSTS"; then
  sudo -u "$UPLINK_USER" ssh-keyscan -H github.com 2>/dev/null \
    >> "$KNOWN_HOSTS"
  chown "${UPLINK_USER}:${UPLINK_USER}" "$KNOWN_HOSTS"
  ok "github.com host key trusted"
fi

# Verify the deploy key works against GitHub
GITHUB_TEST=$(sudo -u "$UPLINK_USER" ssh -o BatchMode=yes \
  -o StrictHostKeyChecking=no -T git@github.com 2>&1 || true)
if echo "$GITHUB_TEST" | grep -q "successfully authenticated"; then
  ok "deploy key works against GitHub"
else
  cat <<EOF

============================================================================
ACTION REQUIRED: Add this deploy key to the GitHub repo, then re-run script
============================================================================

1. Go to:
   https://github.com/ArgusQuote/ElectricalDiagramAnalyzer/settings/keys

2. Click "Add deploy key"
   - Title: argus-prod-aws-${UPLINK_USER} ($(date +%Y-%m-%d))
   - Key: (paste the value below, INCLUDING the 'ssh-ed25519' prefix
           and the trailing comment)
   - Allow write access: Marco's call. The 2026-05-23 redeploy enabled
     write so the AWS box could git push; the original 2026-05-20 key
     was read-only. Either is fine.

3. Click "Add key"

4. Re-run this script on the EC2 box:
       sudo bash $0

----------------------------- PUBLIC KEY -----------------------------------
$(cat "${DEPLOY_KEY}.pub")
----------------------------------------------------------------------------

EOF
  exit 10
fi

# -----------------------------------------------------------------------------
# Step 3: Clone repo (branch TOOL_DEVELOPMENT_V3_MS -- matches production)
# -----------------------------------------------------------------------------
log "Step 3/8: git clone (branch ${REPO_BRANCH})"

if [ ! -d "$REPO_DIR/.git" ]; then
  sudo -u "$UPLINK_USER" git clone --branch "$REPO_BRANCH" \
    "$REPO_URL" "$REPO_DIR"
  ok "repo cloned at $(sudo -u "$UPLINK_USER" git -C "$REPO_DIR" rev-parse HEAD)"
else
  CURRENT_BRANCH=$(sudo -u "$UPLINK_USER" git -C "$REPO_DIR" \
    rev-parse --abbrev-ref HEAD)
  if [ "$CURRENT_BRANCH" != "$REPO_BRANCH" ]; then
    warn "repo on branch '${CURRENT_BRANCH}', expected '${REPO_BRANCH}'"
    warn "switch manually with: cd ${REPO_DIR} && git checkout ${REPO_BRANCH}"
  else
    skip "repo present on branch ${REPO_BRANCH}"
  fi
fi

# -----------------------------------------------------------------------------
# Step 4: Python venv (3.10, matches Paperspace)
# -----------------------------------------------------------------------------
log "Step 4/8: Python venv at ${VENV_DIR}"

if [ ! -d "$VENV_DIR" ]; then
  sudo -u "$UPLINK_USER" python3.10 -m venv "$VENV_DIR"
  sudo -u "$UPLINK_USER" "$VENV_DIR/bin/pip" install --quiet --upgrade \
    pip setuptools wheel
  ok "venv created: $($VENV_DIR/bin/python --version)"
else
  skip "venv exists: $($VENV_DIR/bin/python --version)"
fi

# -----------------------------------------------------------------------------
# Step 5: pip install (uses Paperspace's pip freeze as the canonical spec)
# -----------------------------------------------------------------------------
log "Step 5/8: pip install (this can take ~3-10 minutes; PyTorch + CUDA wheels are large)"

if [ ! -f "$FREEZE_FILE" ]; then
  die "Missing canonical install spec: ${FREEZE_FILE}
This file should have been committed to the repo. Did you clone the right branch?"
fi

# Detect prior successful install via a sentinel file
PIP_DONE_MARKER="${VENV_DIR}/.argus-install-complete"
if [ -f "$PIP_DONE_MARKER" ]; then
  skip "pip install previously completed (sentinel: ${PIP_DONE_MARKER})"
else
  # Strip detectron2 from the freeze before installing. It's a git-source
  # build that compiles CUDA extensions against the local nvcc, which is
  # version 12.0 on this AMI (vs Paperspace's 12.1). Production does NOT
  # use detectron2 -- MLTableDetection/TableDetectorML.py defaults to the
  # "table-transformer" backend and only imports detectron2 lazily inside
  # _load_detectron2_model(). Skipping it here saves ~20 min of compile
  # time and avoids a real risk of failure on CUDA-version mismatch.
  # If detectron2 is ever needed on AWS, install per
  # https://detectron2.readthedocs.io/en/latest/tutorials/install.html
  TEMP_FREEZE=$(mktemp)
  grep -v -E "^detectron2" "$FREEZE_FILE" > "$TEMP_FREEZE"
  STRIPPED=$(($(wc -l < "$FREEZE_FILE") - $(wc -l < "$TEMP_FREEZE")))
  echo "  (excluded ${STRIPPED} package(s) from freeze: detectron2)"

  # mktemp creates the file owned by root with mode 600. Since pip
  # runs as ${UPLINK_USER} via sudo -u, hand the file over so
  # pip can read it (otherwise the install dies with EACCES).
  chown "${UPLINK_USER}:${UPLINK_USER}" "$TEMP_FREEZE"

  # +cu121 PyTorch wheels live at the PyTorch index, not PyPI.
  sudo -u "$UPLINK_USER" "$VENV_DIR/bin/pip" install \
    --extra-index-url https://download.pytorch.org/whl/cu121 \
    -r "$TEMP_FREEZE"
  rm "$TEMP_FREEZE"

  sudo -u "$UPLINK_USER" touch "$PIP_DONE_MARKER"
  ok "pip install complete"
fi

# Verify the CUDA stack is wired up
log "  verifying CUDA via PyTorch"
CUDA_CHECK=$(sudo -u "$UPLINK_USER" "$VENV_DIR/bin/python" -c \
  "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO-GPU')" \
  2>&1 || true)
echo "  -> $CUDA_CHECK"
if ! echo "$CUDA_CHECK" | grep -q "True"; then
  warn "PyTorch cannot see the GPU. Inspect with: source ${VENV_DIR}/bin/activate && python -c 'import torch; print(torch.cuda.is_available())'"
fi

# -----------------------------------------------------------------------------
# Step 6: v7 TATR weights (will be rsynced from your laptop)
# -----------------------------------------------------------------------------
log "Step 6/8: v7 TATR weights"

V7_DIR="${UPLINK_HOME}/Documents/TableAnnotations/models_v7/best"
if [ -f "${V7_DIR}/model.safetensors" ]; then
  V7_SIZE=$(du -sh "$V7_DIR" | cut -f1)
  ok "v7 weights present (${V7_SIZE})"
else
  mkdir -p "$V7_DIR"
  chown -R "${UPLINK_USER}:${UPLINK_USER}" \
    "${UPLINK_HOME}/Documents"
  warn "v7 weights NOT YET PRESENT at ${V7_DIR}"
  warn "From your laptop, rsync them via your workstation (uses your existing ssh access to both hosts):"
  cat <<EOF

  # Two-step transfer (laptop acts as a bridge):
  rsync -avhP paperspace-vm:/home/paperspace/Documents/TableAnnotations/models_v7/best/ \\
      /tmp/v7-weights-cache/
  rsync -avhP /tmp/v7-weights-cache/ \\
      ${UPLINK_USER}@<aws-host>:/tmp/v7-weights/
  # Then on AWS:
  sudo mkdir -p ${V7_DIR}
  sudo mv /tmp/v7-weights/* ${V7_DIR}/
  sudo chown -R ${UPLINK_USER}:${UPLINK_USER} ${UPLINK_HOME}/Documents
  rm -rf /tmp/v7-weights /tmp/v7-weights-cache

EOF
fi

# -----------------------------------------------------------------------------
# Step 7: env file placeholder (you'll paste the real key separately)
# -----------------------------------------------------------------------------
log "Step 7/8: env file placeholder at ${ENV_FILE}"

if [ ! -f "$ENV_FILE" ]; then
  cat > "$ENV_FILE" <<'EOF'
ANVIL_UPLINK_KEY=PASTE_KEY_HERE
EOF
  chown "${UPLINK_USER}:${UPLINK_USER}" "$ENV_FILE"
  chmod 600 "$ENV_FILE"
  ok "${ENV_FILE} created with placeholder (chmod 600, owned by ${UPLINK_USER})"
elif grep -q "PASTE_KEY_HERE" "$ENV_FILE"; then
  warn "${ENV_FILE} still has placeholder. Edit with: sudo nano ${ENV_FILE}"
else
  ok "${ENV_FILE} present (key value not inspected)"
fi

# -----------------------------------------------------------------------------
# Step 8: systemd unit (mirrors Paperspace's anvil-uplink.service)
# -----------------------------------------------------------------------------
log "Step 8/8: systemd unit at ${SYSTEMD_UNIT}"

if [ ! -f "$SYSTEMD_UNIT" ]; then
  cat > "$SYSTEMD_UNIT" <<EOF
[Unit]
Description=Argus Automated BOM -- Anvil Uplink (AWS)
Wants=network-online.target
After=network-online.target

[Service]
Type=simple
User=${UPLINK_USER}
Group=${UPLINK_USER}
WorkingDirectory=${REPO_DIR}
EnvironmentFile=${ENV_FILE}
Environment=PYTHONUNBUFFERED=1
ExecStart=${VENV_DIR}/bin/python ${REPO_DIR}/AnvilUplinkCode/uplink_server.py
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
  systemctl daemon-reload
  ok "systemd unit installed and reloaded (NOT enabled or started -- manual final step)"
else
  skip "systemd unit exists"
fi

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
cat <<EOF

============================================================================
Provisioning complete on $(hostname) at $(date)

Service status: $(systemctl is-enabled argus-uplink 2>/dev/null || echo 'disabled (CORRECT -- never enable this)')

============================================================================
*** READ THIS BEFORE DOING ANYTHING ELSE ***

This box is a DEV / MANUAL-RUN environment. Paperspace is production.

DO NOT run:
  sudo systemctl enable  argus-uplink     <- never
  sudo systemctl enable --now argus-uplink <- never
  sudo systemctl start  argus-uplink      <- never (use manual run below)

The systemd unit exists only so future config can be mirrored if needed.
Running the uplink via systemd makes Anvil round-robin jobs between AWS
and Paperspace, which is the OPPOSITE of the current goal (Paperspace
serves 100% of customer traffic; AWS only runs when Marco is actively
testing in a controlled session).
============================================================================

Manual finishing steps (in order):

1. If v7 TATR weights aren't there yet, rsync them from Paperspace
   (see Step 6 message above for the exact commands).

2. Set the ANVIL_UPLINK_KEY in ${ENV_FILE}:
   sudo nano ${ENV_FILE}
   # Replace 'PASTE_KEY_HERE' with the value from Paperspace's
   # /home/paperspace/.anvil_env. Save (Ctrl-O, Enter, Ctrl-X).

3. Verify the box is ready (does NOT start the uplink; safe at any time):
   source ${VENV_DIR}/bin/activate
   cd ${REPO_DIR}
   python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   # expect: True NVIDIA A10G
   python -c "import anvil.server, easyocr, transformers; print('imports ok')"

4. To run the uplink for a manual development session (only when Marco
   explicitly wants to test; this WILL connect to Anvil and cause it to
   route ~half of new jobs to AWS until Ctrl-C):
       source ${VENV_DIR}/bin/activate
       cd ${REPO_DIR}
       export \$(cat ${ENV_FILE} | xargs)
       python AnvilUplinkCode/uplink_server.py
       # ... test, then Ctrl-C to stop and disconnect from Anvil.

============================================================================
EOF
