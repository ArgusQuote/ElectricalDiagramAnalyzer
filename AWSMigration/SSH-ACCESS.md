# SSH access to the Argus AWS dev box

The AWS dev box (**Argus Server**, `g6.2xlarge` / NVIDIA L4) is reachable only
through **AWS Systems Manager (SSM) Session Manager**. Port 22 is closed on the
public internet; your laptop opens a tunnel through AWS, then SSH authenticates
with the EC2 key pair.

This is a **manual-run development box**, not customer production. Paperspace
continues to serve live Anvil traffic. Do not enable `argus-uplink.service` on
AWS.

## Two layers of access

| Layer | What it is | Who provides it |
|-------|------------|-----------------|
| **AWS IAM** | Permission to start an SSM session to the instance | Marco creates an IAM user and sends access keys securely |
| **EC2 key** (`argus-prod-key.pem`) | Linux login as `ubuntu` | Marco sends the private key file securely (never commit it) |

Both are required. Cursor needs no special configuration — use a normal terminal
and run `ssh argus-prod-aws`.

## Fixed connection details

| Setting | Value |
|---------|--------|
| Instance ID | `i-05a721f0f6b122179` |
| Region | `us-east-1` |
| Linux user | `ubuntu` |
| SSH alias | `argus-prod-aws` |
| IAM user (current guest) | `argus-ssh-friend` |

The instance must be **running** and SSM **Online** before SSH will work. Marco
starts it from the EC2 Console when needed; stopped instances save compute
cost (~$1+/hr while running).

---

## First-time setup (Linux)

After Marco sends your AWS access keys and `argus-prod-key.pem`:

```bash
git pull   # in your ElectricalDiagramAnalyzer clone
cp /path/from/marco/argus-prod-key.pem ~/.ssh/argus-prod-key.pem
chmod 600 ~/.ssh/argus-prod-key.pem

bash AWSMigration/setup-ssh-access.sh
# Enter Access Key ID, Secret Access Key, region us-east-1 when prompted

ssh argus-prod-aws
```

The setup script installs AWS CLI v2 and the Session Manager plugin (user-local,
no `sudo`), runs `aws configure` if needed, and appends the SSH config block.

### Manual Linux setup

If you prefer not to run the script:

1. Install [AWS CLI v2](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)
2. Install the [Session Manager plugin](https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager-working-with-install-plugin.html) (Ubuntu `.deb` or equivalent)
3. Run `aws configure` — region **`us-east-1`**
4. Add to `~/.ssh/config`:

```
Host argus-prod-aws
    HostName i-05a721f0f6b122179
    User ubuntu
    IdentityFile ~/.ssh/argus-prod-key.pem
    IdentitiesOnly yes
    ProxyCommand sh -c "PATH=$HOME/.local/bin:$PATH exec aws ssm start-session --target %h --document-name AWS-StartSSHSession --parameters portNumber=%p"
```

Adjust the `PATH=` prefix if `aws` and `session-manager-plugin` live elsewhere
(e.g. `/usr/local/bin`).

---

## First-time setup (macOS)

1. Install AWS CLI v2 (`brew install awscli` or the official pkg installer)
2. Install the Session Manager plugin:
   ```bash
   brew install --cask session-manager-plugin
   ```
   Or download the macOS pkg from the [AWS plugin page](https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager-working-with-install-plugin.html).
3. Save Marco's `argus-prod-key.pem` to `~/.ssh/argus-prod-key.pem` and `chmod 600` it
4. Run `aws configure` with the keys Marco sent — region **`us-east-1`**
5. Add the same `Host argus-prod-aws` block above to `~/.ssh/config` (often
   `PATH` can omit the prefix if both tools are on your default PATH)
6. Connect: `ssh argus-prod-aws`

---

## Verify before SSH

```bash
aws sts get-caller-identity
# Expect Arn containing user/argus-ssh-friend (or your assigned IAM user)

aws ssm describe-instance-information \
  --region us-east-1 \
  --filters "Key=InstanceIds,Values=i-05a721f0f6b122179" \
  --query 'InstanceInformationList[0].PingStatus' --output text
# Expect: Online  (if instance is running)
```

---

## Daily use

```bash
ssh argus-prod-aws
source /home/ubuntu/venv/bin/activate
cd /home/ubuntu/ElectricalDiagramAnalyzer
```

Copy files with `scp`:

```bash
scp localfile argus-prod-aws:/tmp/
```

---

## Troubleshooting

| Symptom | Likely fix |
|---------|------------|
| `TargetNotConnected` / no SSM entry | Instance stopped — ask Marco to start it in EC2 Console |
| `AccessDeniedException` on `start-session` | Wrong/expired IAM keys, or user not attached to `ArgusSSMSSHOnly` policy |
| `session-manager-plugin not found` | Install plugin; ensure `PATH` in ProxyCommand includes its directory |
| SSH "Permission denied (publickey)" | Wrong or missing `~/.ssh/argus-prod-key.pem`, or permissions not `600` |
| Hangs then times out | Instance still booting — wait ~2 min for status checks |

---

## For Marco: grant or revoke access

### IAM policy (already created)

Policy name: **`ArgusSSMSSHOnly`**

Allows SSM SSH to instance `i-05a721f0f6b122179` only, plus describe instance
status. Does not allow starting/stopping EC2, changing security groups, or
accessing other AWS resources.

### Grant a new user

```bash
aws iam create-user --user-name <username>
aws iam attach-user-policy \
  --user-name <username> \
  --policy-arn arn:aws:iam::766851565012:policy/ArgusSSMSSHOnly
aws iam create-access-key --user-name <username>
```

Send the access key and `argus-prod-key.pem` through a secure channel (Signal,
1Password, etc.). **Never commit keys to this repo.**

Point them at this file and `AWSMigration/setup-ssh-access.sh`.

### Revoke access

```bash
aws iam list-access-keys --user-name argus-ssh-friend
aws iam delete-access-key --user-name argus-ssh-friend --access-key-id <KEY_ID>
aws iam detach-user-policy \
  --user-name argus-ssh-friend \
  --policy-arn arn:aws:iam::766851565012:policy/ArgusSSMSSHOnly
aws iam delete-user --user-name argus-ssh-friend
```

---

## Security notes

- Guest users share the **`ubuntu`** Linux account, including the GitHub deploy
  key on the box (write-enabled). Treat access as fully trusted.
- Revoke IAM keys when access is no longer needed; sharing the same
  `argus-prod-key.pem` does not grant SSM access without valid AWS credentials.
- Do not run `python AnvilUplinkCode/uplink_server.py` unless Marco explicitly
  asks — it round-robins live customer jobs with Paperspace until stopped.
