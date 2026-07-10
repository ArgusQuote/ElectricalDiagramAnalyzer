#!/usr/bin/env bash
# One-time laptop setup for SSH to the Argus AWS dev box via SSM.
# See AWSMigration/SSH-ACCESS.md for full instructions.

set -euo pipefail

echo "=== Argus AWS SSH setup ==="

install_dir="${HOME}/.local/bin"
mkdir -p "${install_dir}"
export PATH="${install_dir}:${PATH}"

if ! command -v aws >/dev/null 2>&1; then
  echo "Installing AWS CLI v2..."
  if [[ "$(uname -s)" == "Darwin" ]]; then
    echo "On macOS, install AWS CLI first: brew install awscli"
    exit 1
  fi
  curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip
  unzip -q /tmp/awscliv2.zip -d /tmp
  /tmp/aws/install --install-dir "${HOME}/.local/aws-cli" --bin-dir "${install_dir}"
fi

if ! command -v session-manager-plugin >/dev/null 2>&1; then
  echo "Installing Session Manager plugin..."
  if [[ "$(uname -s)" == "Darwin" ]]; then
    if command -v brew >/dev/null 2>&1; then
      brew install --cask session-manager-plugin
    else
      echo "Install the macOS Session Manager plugin manually — see AWSMigration/SSH-ACCESS.md"
      exit 1
    fi
  else
    curl -fsSL "https://s3.amazonaws.com/session-manager-downloads/plugin/latest/ubuntu_64bit/session-manager-plugin.deb" \
      -o /tmp/session-manager-plugin.deb
    mkdir -p /tmp/ssm-plugin
    dpkg-deb -x /tmp/session-manager-plugin.deb /tmp/ssm-plugin
    cp /tmp/ssm-plugin/usr/local/sessionmanagerplugin/bin/session-manager-plugin "${install_dir}/"
    chmod +x "${install_dir}/session-manager-plugin"
  fi
fi

mkdir -p "${HOME}/.ssh"
chmod 700 "${HOME}/.ssh"

if [[ ! -f "${HOME}/.ssh/argus-prod-key.pem" ]]; then
  echo "ERROR: Put argus-prod-key.pem in ~/.ssh/argus-prod-key.pem first (Marco sends separately)."
  exit 1
fi
chmod 600 "${HOME}/.ssh/argus-prod-key.pem"

if [[ ! -f "${HOME}/.aws/credentials" ]]; then
  echo ""
  echo "Configure AWS (Marco sends Access Key ID + Secret separately):"
  aws configure
  echo "Use region: us-east-1"
fi

ssh_config="${HOME}/.ssh/config"
if ! grep -q 'Host argus-prod-aws' "${ssh_config}" 2>/dev/null; then
  cat >> "${ssh_config}" << 'EOF'

Host argus-prod-aws
    HostName i-05a721f0f6b122179
    User ubuntu
    IdentityFile ~/.ssh/argus-prod-key.pem
    IdentitiesOnly yes
    ProxyCommand sh -c "PATH=$HOME/.local/bin:$PATH exec aws ssm start-session --target %h --document-name AWS-StartSSHSession --parameters portNumber=%p"
EOF
  chmod 600 "${ssh_config}"
  echo "Added argus-prod-aws to ~/.ssh/config"
else
  echo "~/.ssh/config already has argus-prod-aws"
fi

echo ""
echo "Verifying AWS identity..."
aws sts get-caller-identity
echo ""
echo "Checking SSM (instance must be running)..."
aws ssm describe-instance-information \
  --region us-east-1 \
  --filters "Key=InstanceIds,Values=i-05a721f0f6b122179" \
  --query 'InstanceInformationList[0].PingStatus' --output text
echo ""
echo "Try: ssh argus-prod-aws"
