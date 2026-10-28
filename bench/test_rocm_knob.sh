#!/bin/bash
# CPU test for ROCM knob (launchers and resolve_config.sh).
# Copies checkout to a temp dir and runs the three launchers against a stub venv/bin/vllm.
# No GPU, no model, no network.
#
#   bash bench/test_rocm_knob.sh
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(dirname "$HERE")"
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
(cd "$REPO" && tar --exclude=.git --exclude=./venv --exclude=docs/media --exclude=bench/demo --exclude=models -cf - .) | tar -xf - -C "$T"
mkdir -p "$T/venv/bin" "$T/models/Qwen3.8-27B-W4A16-AutoRound" "$T/models/Qwen3.8-27B-DFlash2-W4A16"
: > "$T/models/Qwen3.8-27B-W4A16-AutoRound/model.safetensors"
: > "$T/models/Qwen3.8-27B-DFlash2-W4A16/model.safetensors"

printf '#!/bin/sh
echo "@@ENV"
echo "VLLM_SPEC_DECODE_ATTN=${VLLM_SPEC_DECODE_ATTN-unset}"
echo "KVARN_FP16_DEQUANT=${KVARN_FP16_DEQUANT-unset}"
echo "VLLM_MARLIN_INPUT_DTYPE=${VLLM_MARLIN_INPUT_DTYPE-unset}"
env | grep -E "^(VLLM_|KVARN_)" | sort
echo "@@ARGV"
printf "%%s\\n" "$@"
' > "$T/venv/bin/vllm"
chmod +x "$T/venv/bin/vllm"

FAILS=0
check() { local want=$1 got=$2 what=$3
  if [ "$got" = "$want" ]; then
    printf '  PASS  %s\n' "$what"
  else
    printf '  FAIL  %s -> %s (want %s)\n' "$what" "${got:-<none>}" "$want"
    FAILS=$((FAILS+1))
  fi
}

run() { local sc=$1; shift
  rm -f "$T/out" "$T/err"
  (cd "$T" && env -u PRINT_ARGV \
    -u ROCM -u CTX -u SPEC -u KV -u INT8_ACT -u INT8_LAYERS \
    -u PTH_FA -u FA2_FP8KV -u KVARN_FP16_DEQUANT \
    -u VLLM_SPEC_DECODE_ATTN -u VLLM_MARLIN_INPUT_DTYPE \
    VLLM_API_KEY=testkey "$@" bash "$sc" >"$T/out" 2>"$T/err" </dev/null)
  echo $?
}

env_val() { grep -m1 "^$1=" "$T/out" | cut -d= -f2-; }
arg_after() { sed -n "/^$1\$/{n;p;}" "$T/out"; }
err_has() { grep -qF -- "$1" "$T/err" && echo yes || echo no; }

echo "== ROCM=invalid: refused by all launchers and resolve_config.sh"
check 1 "$(run single-user/start_qwen.sh ROCM=invalid)" "single-user/start_qwen.sh: ROCM=invalid refused"
check yes "$(err_has 'ROCM=invalid')" "  refusal message contains ROCM=invalid"

check 1 "$(run batch/start_qwen.sh ROCM=invalid)" "batch/start_qwen.sh: ROCM=invalid refused"
check yes "$(err_has 'ROCM=invalid')" "  refusal message contains ROCM=invalid"

check 1 "$(run single-user/alternative.sh ROCM=invalid)" "single-user/alternative.sh: ROCM=invalid refused"
check yes "$(err_has 'ROCM=invalid')" "  refusal message contains ROCM=invalid"

rm -f "$T/out" "$T/err"
(cd "$T" && env REPO="$T" ROCM=invalid bash resolve_config.sh single >"$T/out" 2>"$T/err")
rc=$?
check 1 "$rc" "resolve_config.sh single: ROCM=invalid refused"
check yes "$(err_has 'ROCM=invalid')" "  refusal message contains ROCM=invalid"

echo "== single-user/start_qwen.sh: ROCM=1 CTX=fast"
check 0 "$(run single-user/start_qwen.sh ROCM=1 CTX=fast)" "CTX=fast: starts"
check "TRITON_ATTN" "$(arg_after --attention-backend)" "  --attention-backend TRITON_ATTN"
check "bfloat16" "$(arg_after --kv-cache-dtype)" "  --kv-cache-dtype bfloat16"
check "1" "$(env_val VLLM_SPEC_DECODE_ATTN)" "  VLLM_SPEC_DECODE_ATTN=1"

echo "== single-user/start_qwen.sh: ROCM=1 SPEC=mtp CTX=long"
check 0 "$(run single-user/start_qwen.sh ROCM=1 SPEC=mtp CTX=long)" "SPEC=mtp CTX=long: starts"
check "TRITON_ATTN" "$(arg_after --attention-backend)" "  --attention-backend TRITON_ATTN"
check "int8_per_token_head" "$(arg_after --kv-cache-dtype)" "  --kv-cache-dtype int8_per_token_head"
check "1" "$(env_val VLLM_SPEC_DECODE_ATTN)" "  VLLM_SPEC_DECODE_ATTN=1"

echo "== single-user/start_qwen.sh: ROCM=1 SPEC=dflash2 CTX=long"
check 0 "$(run single-user/start_qwen.sh ROCM=1 SPEC=dflash2 CTX=long)" "SPEC=dflash2 CTX=long: starts"
check "TRITON_ATTN" "$(arg_after --attention-backend)" "  --attention-backend TRITON_ATTN"
check "int8_per_token_head" "$(arg_after --kv-cache-dtype)" "  --kv-cache-dtype int8_per_token_head"
check "1" "$(env_val VLLM_SPEC_DECODE_ATTN)" "  VLLM_SPEC_DECODE_ATTN=1"

echo "== single-user/start_qwen.sh: ROCM=1 CTX=huge (KVarN dequant)"
check 0 "$(run single-user/start_qwen.sh ROCM=1 CTX=huge)" "CTX=huge: starts"
check "1" "$(env_val KVARN_FP16_DEQUANT)" "  KVARN_FP16_DEQUANT=1 by default"
check 0 "$(run single-user/start_qwen.sh ROCM=1 CTX=huge KVARN_FP16_DEQUANT=0)" "CTX=huge KVARN_FP16_DEQUANT=0: starts"
check "0" "$(env_val KVARN_FP16_DEQUANT)" "  KVARN_FP16_DEQUANT=0 explicit respected"

echo "== single-user/start_qwen.sh: ROCM=1 INT8_ACT=int8"
check 0 "$(run single-user/start_qwen.sh ROCM=1 INT8_ACT=int8)" "INT8_ACT=int8: starts"
check "unset" "$(env_val VLLM_MARLIN_INPUT_DTYPE)" "  VLLM_MARLIN_INPUT_DTYPE stays unset"
check yes "$(err_has 'Marlin W4A8 is CUDA-only')" "  stderr note about Marlin W4A8 being CUDA-only"

echo "== single-user/start_qwen.sh: ROCM=1 refusals (PTH_FA=1, FA2_FP8KV=1)"
check 1 "$(run single-user/start_qwen.sh ROCM=1 PTH_FA=1)" "PTH_FA=1: refused"
check yes "$(err_has 'ROCM=1 does not support PTH_FA=1')" "  reason mentions PTH_FA=1 unsupported"
check 1 "$(run single-user/start_qwen.sh ROCM=1 FA2_FP8KV=1)" "FA2_FP8KV=1: refused"
check yes "$(err_has 'ROCM=1 does not support FA2_FP8KV=1')" "  reason mentions FA2_FP8KV=1 unsupported"

echo "== batch/start_qwen.sh: ROCM=1"
check 0 "$(run batch/start_qwen.sh ROCM=1)" "default KV: starts"
check "int4_per_token_head" "$(arg_after --kv-cache-dtype)" "  --kv-cache-dtype int4_per_token_head"
check "TRITON_ATTN" "$(arg_after --attention-backend)" "  --attention-backend TRITON_ATTN"
check "unset" "$(env_val VLLM_MARLIN_INPUT_DTYPE)" "  VLLM_MARLIN_INPUT_DTYPE stays unset"

check 1 "$(run batch/start_qwen.sh ROCM=1 KV=fp8)" "KV=fp8: refused"
check yes "$(err_has 'ROCM=1 does not support KV=fp8')" "  reason mentions KV=fp8 unsupported"

check 0 "$(run batch/start_qwen.sh ROCM=1 KV=kvarn)" "KV=kvarn: starts"
check "1" "$(env_val KVARN_FP16_DEQUANT)" "  KVARN_FP16_DEQUANT=1 exported"

echo
[ $FAILS = 0 ] && echo "all checks passed" || { echo "$FAILS checks FAILED"; exit 1; }
