#!/bin/bash
# Which container isolation each weight-cache transport can survive.
#
# Runs both probes across a grid of docker isolation settings. Requires docker
# and at least one GPU on the host; nothing is allocated on GPUs other than 0.
#
#   bash transport_matrix.sh [image]
#
# Reads as: the shipped torch_ipc transport needs a shared IPC namespace AND a
# shared PID namespace (each alone fails, with different driver errors), while
# vmm_fd needs neither -- only the shared bind mount that carries its socket.
set -u

IMG="${1:-lmsysorg/sglang:latest}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARE="${SHARE_DIR:-/tmp/wc_transport_matrix}"
RESULTS="$SHARE/../wc_transport_matrix.txt"
: > "$RESULTS"

NV="--runtime=nvidia -e NVIDIA_DRIVER_CAPABILITIES=compute,utility"
NV="$NV -e NVIDIA_VISIBLE_DEVICES=0 -e CUDA_VISIBLE_DEVICES=0"
MNT="-v $SHARE:/shared -v $HERE:/probe:ro"

cleanup() {
  docker rm -f wcm_exp wcm_imp >/dev/null 2>&1
  rm -rf "$SHARE"; mkdir -p "$SHARE"
}

# wait_for <path-under-$SHARE> <seconds>
wait_for() {
  local i
  for ((i = 0; i < $2; i++)); do
    [ -e "$SHARE/$1" ] && return 0
    sleep 1
  done
  return 1
}

# probe <label> <transport: torch_ipc|vmm_fd> <docker flags for both sides>
probe() {
  local label="$1" transport="$2" flags="$3"
  local script ready arg
  if [ "$transport" = "vmm_fd" ]; then
    script=vmm_fd_probe.py; ready=vmm.sock; arg=/shared/vmm.sock
  else
    script=torch_ipc_probe.py; ready=ready; arg=/shared
  fi

  cleanup
  docker run -d --rm --name wcm_exp $NV $flags $MNT \
    --entrypoint /bin/bash "$IMG" -c "python /probe/$script export $arg" \
    >/dev/null 2>&1
  if ! wait_for "$ready" 120; then
    printf '%-34s %-9s | EXPORTER_NEVER_READY %s\n' "$label" "$transport" \
      "$(docker logs wcm_exp 2>&1 | tail -2 | tr '\n' ' ')" >> "$RESULTS"
    cleanup; return
  fi

  local out
  out=$(docker run --rm --name wcm_imp $NV $flags $MNT \
    --entrypoint /bin/bash "$IMG" -c "python /probe/$script import $arg" 2>&1 \
    | grep -E 'RESULT=' | tr '\n' ' ')
  printf '%-34s %-9s | %s\n' "$label" "$transport" "${out:-NO_OUTPUT}" >> "$RESULTS"
  cleanup
}

for transport in torch_ipc vmm_fd; do
  echo "=== $transport"
  probe "no namespace sharing"      "$transport" ""
  probe "ipc=host only"             "$transport" "--ipc=host"
  probe "pid=host only"             "$transport" "--pid=host"
  probe "ipc=host + pid=host"       "$transport" "--ipc=host --pid=host"
done

echo
echo "############ transport x isolation ############"
cat "$RESULTS"
rm -rf "$SHARE"
