#!/usr/bin/env bash
# Put this computer's SSH public key on a server, test the key login, and add a short alias to ~/.ssh/config.
# Usage:  bash ssh_setup.sh HOST PORT [USER] [ALIAS] [KEYFILE]
#   bash ssh_setup.sh connect.bjb1.seetacloud.com 54135 root gpu1
# The server's password is typed into ssh itself (once, when the key is installed); it is never stored.
set -euo pipefail

HOST=${1:?usage: ssh_setup.sh HOST PORT [USER] [ALIAS] [KEYFILE]}
PORT=${2:?usage: ssh_setup.sh HOST PORT [USER] [ALIAS] [KEYFILE]}
USER_=${3:-root}
ALIAS=${4:-}
KEY=${5:-$HOME/.ssh/id_rsa}

[ -f "$KEY" ] && [ -f "$KEY.pub" ] || { echo "no key pair at $KEY(.pub); create one with: ssh-keygen -t ed25519"; exit 1; }
SSH=(ssh -p "$PORT" -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new)

echo "== 1/3 key login already works?"
if "${SSH[@]}" -i "$KEY" -o BatchMode=yes -o PasswordAuthentication=no "$USER_@$HOST" true 2>/dev/null; then
  echo "yes, nothing to install"
else
  echo "no: installing $KEY.pub (type the server password when asked)"
  "${SSH[@]}" -o PreferredAuthentications=password,keyboard-interactive -o PubkeyAuthentication=no "$USER_@$HOST" \
    'umask 077; mkdir -p ~/.ssh; touch ~/.ssh/authorized_keys; chmod 700 ~/.ssh; chmod 600 ~/.ssh/authorized_keys;
     read -r k; grep -qxF "$k" ~/.ssh/authorized_keys || echo "$k" >> ~/.ssh/authorized_keys' < "$KEY.pub"
fi

echo "== 2/3 test the key login"
"${SSH[@]}" -i "$KEY" -o BatchMode=yes -o PasswordAuthentication=no "$USER_@$HOST" \
  'echo "login OK on $(hostname)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "no nvidia-smi"; df -h / | tail -n 1'

if [ -n "$ALIAS" ]; then
  echo "== 3/3 ssh alias '$ALIAS'"
  CFG="$HOME/.ssh/config"; mkdir -p "$HOME/.ssh"; touch "$CFG"
  if grep -qE "^Host[[:space:]]+$ALIAS([[:space:]]|$)" "$CFG"; then
    echo "alias already in $CFG, left as it is"
  else
    printf '\nHost %s\n  HostName %s\n  Port %s\n  User %s\n  IdentityFile %s\n  ServerAliveInterval 30\n' \
      "$ALIAS" "$HOST" "$PORT" "$USER_" "$KEY" >> "$CFG"
    echo "added; connect with: ssh $ALIAS"
  fi
else
  echo "== 3/3 no alias requested; connect with: ssh -p $PORT $USER_@$HOST"
fi
