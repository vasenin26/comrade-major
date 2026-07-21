#!/usr/bin/env bash
# Подготовка системного окружения для voice-agent.
# Python-пакеты ставятся в систему (без venv) — проект рассчитан на выделенную машину.
# Запуск: sudo ./scripts/setup-environment.sh
#
# Переменные окружения:
#   FORCE_CPU=1          — принудительно CPU-сборка PyTorch
#   TORCH_CUDA_INDEX=... — переопределить index-url PyTorch (например cu126)

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MIN_PYTHON_VERSION="3.12"
TORCH_CPU_INDEX="https://download.pytorch.org/whl/cpu"

BASE_APT_PACKAGES=(
    ca-certificates
    curl
    git
    build-essential
    pkg-config
    portaudio19-dev
    libportaudio2
    libsndfile1
    ffmpeg
)

CUDA_AVAILABLE=0
GPU_NAME=""
DRIVER_VERSION=""
CUDA_DRIVER_VERSION=""
TORCH_INDEX="${TORCH_CPU_INDEX}"

log() {
    printf '[setup] %s\n' "$*"
}

warn() {
    printf '[setup][warn] %s\n' "$*" >&2
}

die() {
    printf '[setup][error] %s\n' "$*" >&2
    exit 1
}

require_root() {
    if [[ "${EUID:-$(id -u)}" -ne 0 ]]; then
        die "Скрипт нужно запускать от администратора: sudo ./scripts/setup-environment.sh"
    fi
}

resolve_target_user() {
    if [[ -n "${SUDO_USER:-}" && "${SUDO_USER}" != "root" ]]; then
        TARGET_USER="${SUDO_USER}"
    else
        TARGET_USER="${USER:-root}"
    fi

    if [[ "${TARGET_USER}" == "root" ]]; then
        warn "SUDO_USER не определён — .env будет создан от root."
        TARGET_UID="$(id -u)"
        TARGET_GID="$(id -g)"
        return 0
    fi

    TARGET_UID="$(id -u "${TARGET_USER}")"
    TARGET_GID="$(id -g "${TARGET_USER}")"
}

have_command() {
    command -v "$1" >/dev/null 2>&1
}

apt_package_available() {
    apt-cache show "$1" >/dev/null 2>&1
}

install_missing_packages() {
    local missing=()
    local package

    for package in "$@"; do
        if ! dpkg -s "${package}" >/dev/null 2>&1; then
            missing+=("${package}")
        fi
    done

    if ((${#missing[@]} == 0)); then
        return 0
    fi

    log "Устанавливаю пакеты: ${missing[*]}"
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "${missing[@]}"
}

find_python_version_in_apt() {
    local version
    # 3.12 — предпочтительнее для ML-стека; затем более новые версии.
    for version in 3.12 3.13 3.14; do
        if apt_package_available "python${version}"; then
            echo "${version}"
            return 0
        fi
    done
    return 1
}

enable_deadsnakes_ppa() {
    if [[ ! -f /etc/os-release ]]; then
        return 1
    fi

    # shellcheck disable=SC1091
    source /etc/os-release
    if [[ "${ID:-}" != "ubuntu" ]]; then
        return 1
    fi

    log "Подключаю PPA deadsnakes для Python 3.12+ (Ubuntu)..."
    install_missing_packages software-properties-common
    add-apt-repository -y ppa:deadsnakes/ppa
    apt-get update -qq
}

install_python_runtime() {
    PYTHON_VERSION="$(find_python_version_in_apt || true)"

    if [[ -z "${PYTHON_VERSION:-}" ]]; then
        warn "Python 3.12+ не найден в текущих репозиториях."
        enable_deadsnakes_ppa || true
        PYTHON_VERSION="$(find_python_version_in_apt || true)"
    fi

    if [[ -z "${PYTHON_VERSION:-}" ]]; then
        die "Не удалось найти Python ${MIN_PYTHON_VERSION}+ в apt. Добавьте репозиторий вручную (на Ubuntu: deadsnakes PPA)."
    fi

    PYTHON_BIN="python${PYTHON_VERSION}"
    log "Выбран Python ${PYTHON_VERSION} (пакет apt: python${PYTHON_VERSION})"

    local python_packages=(
        "python${PYTHON_VERSION}"
        "python${PYTHON_VERSION}-dev"
        python3-pip
    )

    if apt_package_available "python${PYTHON_VERSION}-venv"; then
        python_packages+=("python${PYTHON_VERSION}-venv")
    fi

    install_missing_packages "${python_packages[@]}"

    if ! have_command "${PYTHON_BIN}"; then
        die "Бинарник ${PYTHON_BIN} не найден после установки пакетов."
    fi
}

install_base_packages() {
    if ! have_command apt-get; then
        die "apt-get не найден. Скрипт рассчитан на Debian/Ubuntu/WSL."
    fi

    log "Обновляю индекс пакетов apt..."
    apt-get update -qq
    install_missing_packages "${BASE_APT_PACKAGES[@]}"
}

resolve_python() {
    log "Использую Python: $(command -v "${PYTHON_BIN}")"
    MIN_PYTHON_VERSION="${MIN_PYTHON_VERSION}" "${PYTHON_BIN}" - <<'PY'
import os
import sys

major, minor = map(int, os.environ["MIN_PYTHON_VERSION"].split("."))
if sys.version_info < (major, minor):
    raise SystemExit(
        f"Требуется Python {os.environ['MIN_PYTHON_VERSION']}+, "
        f"найден {sys.version.split()[0]}"
    )
PY
}

select_pytorch_index() {
    local cuda_major cuda_minor

    if [[ "${FORCE_CPU:-0}" == "1" ]]; then
        warn "FORCE_CPU=1 — использую CPU-сборку PyTorch."
        TORCH_INDEX="${TORCH_CPU_INDEX}"
        return 0
    fi

    if [[ -n "${TORCH_CUDA_INDEX:-}" ]]; then
        TORCH_INDEX="${TORCH_CUDA_INDEX}"
        CUDA_AVAILABLE=1
        log "TORCH_CUDA_INDEX задан вручную: ${TORCH_INDEX}"
        return 0
    fi

    if ! have_command nvidia-smi; then
        warn "nvidia-smi не найден — PyTorch будет установлен в CPU-режиме."
        TORCH_INDEX="${TORCH_CPU_INDEX}"
        return 0
    fi

    if ! nvidia-smi >/dev/null 2>&1; then
        warn "nvidia-smi не отвечает — PyTorch будет установлен в CPU-режиме."
        TORCH_INDEX="${TORCH_CPU_INDEX}"
        return 0
    fi

    CUDA_AVAILABLE=1
    GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1 | xargs)"
    DRIVER_VERSION="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1 | xargs)"
    CUDA_DRIVER_VERSION="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: \([0-9.]*\).*/\1/p' | head -n1)"

    cuda_major="${CUDA_DRIVER_VERSION%%.*}"
    cuda_minor="${CUDA_DRIVER_VERSION#*.}"
    cuda_minor="${cuda_minor%%.*}"

    if [[ "${cuda_major}" -ge 13 ]] || { [[ "${cuda_major}" -eq 12 ]] && [[ "${cuda_minor:-0}" -ge 6 ]]; }; then
        TORCH_INDEX="https://download.pytorch.org/whl/cu126"
    elif [[ "${cuda_major}" -eq 12 ]]; then
        TORCH_INDEX="https://download.pytorch.org/whl/cu124"
    elif [[ "${cuda_major}" -eq 11 ]]; then
        TORCH_INDEX="https://download.pytorch.org/whl/cu118"
    else
        warn "Неизвестная CUDA ${CUDA_DRIVER_VERSION}, использую cu124."
        TORCH_INDEX="https://download.pytorch.org/whl/cu124"
    fi

    log "NVIDIA GPU: ${GPU_NAME}"
    log "Драйвер: ${DRIVER_VERSION}, CUDA (driver): ${CUDA_DRIVER_VERSION}"
    log "PyTorch index: ${TORCH_INDEX}"
}

detect_cuda() {
    select_pytorch_index
}

pip_install_args() {
    PIP_ARGS=()
    local managed_file
    for managed_file in "/usr/lib/${PYTHON_BIN}/EXTERNALLY-MANAGED" "/usr/lib/python3/EXTERNALLY-MANAGED"; do
        if [[ -f "${managed_file}" ]]; then
            PIP_ARGS+=(--break-system-packages)
            log "PEP 668: использую --break-system-packages для системной установки."
            break
        fi
    done
}

ensure_pip() {
    if "${PYTHON_BIN}" -m pip --version >/dev/null 2>&1; then
        return 0
    fi

    log "pip не найден для ${PYTHON_BIN}, пробую ensurepip..."
    "${PYTHON_BIN}" -m ensurepip --upgrade || die "Не удалось установить pip для ${PYTHON_BIN}."
}

install_pytorch() {
    log "Устанавливаю PyTorch/torchaudio (${TORCH_INDEX})..."
    "${PYTHON_BIN}" -m pip install --upgrade torch torchaudio \
        --index-url "${TORCH_INDEX}" \
        "${PIP_ARGS[@]:-}"
}

install_python_dependencies() {
    ensure_pip

    log "Обновляю pip..."
    "${PYTHON_BIN}" -m pip install --upgrade pip "${PIP_ARGS[@]:-}"

    detect_cuda
    install_pytorch

    log "Устанавливаю voice-agent и dev-зависимости..."
    "${PYTHON_BIN}" -m pip install -e "${ROOT_DIR}[dev]" "${PIP_ARGS[@]:-}"
}

prepare_env_file() {
    if [[ -f "${ROOT_DIR}/.env" ]]; then
        log "Файл .env уже существует — пропускаю."
        return 0
    fi

    if [[ ! -f "${ROOT_DIR}/.env.example" ]]; then
        warn ".env.example не найден — создайте .env вручную."
        return 0
    fi

    log "Создаю .env из .env.example..."
    cp "${ROOT_DIR}/.env.example" "${ROOT_DIR}/.env"
    chown "${TARGET_UID}:${TARGET_GID}" "${ROOT_DIR}/.env"
}

verify_imports() {
    "${PYTHON_BIN}" -c 'import sounddevice, transformers, faster_whisper; print("imports: OK")'
}

verify_cuda() {
    log "Проверяю доступность CUDA в PyTorch..."
    VERIFY_CUDA_EXPECTED="${CUDA_AVAILABLE}" "${PYTHON_BIN}" - <<'PY'
import os
import sys

import torch

print(f"PyTorch {torch.__version__}")
print(f"torch.version.cuda = {torch.version.cuda}")

if torch.cuda.is_available():
    count = torch.cuda.device_count()
    print(f"CUDA доступна: {count} GPU")
    for index in range(count):
        props = torch.cuda.get_device_properties(index)
        print(f"  [{index}] {props.name} — {props.total_memory // (1024 ** 2)} MiB")
    tensor = torch.tensor([1.0], device="cuda")
    assert tensor.device.type == "cuda"
    print("CUDA smoke test: OK")
else:
    print("CUDA недоступна — приложение будет работать на CPU.")
    expected = os.environ.get("VERIFY_CUDA_EXPECTED", "0")
    if expected == "1":
        print(
            "WARN: nvidia-smi видит GPU, но torch.cuda.is_available() = False. "
            "Проверьте драйвер/WSL CUDA.",
            file=sys.stderr,
        )
        sys.exit(1)
PY
}

verify_setup() {
    log "Проверяю установку..."
    verify_imports
    verify_cuda
    (cd "${ROOT_DIR}" && "${PYTHON_BIN}" -m pytest -q)
}

print_summary() {
    local cuda_note="CPU"
    if [[ "${CUDA_AVAILABLE}" == "1" ]]; then
        cuda_note="GPU (${GPU_NAME:-NVIDIA})"
    fi

    cat <<EOF

Готово. Зависимости установлены в системный Python (${PYTHON_BIN}).
Режим PyTorch: ${cuda_note}
Index URL: ${TORCH_INDEX}

Дальнейшие шаги:
  cd ${ROOT_DIR}
  cp .env.example .env   # если .env ещё не создан
  ${PYTHON_BIN} -m src.main

Проверки:
  ${PYTHON_BIN} -m pytest
  ${PYTHON_BIN} -m mypy config src

Переопределение:
  FORCE_CPU=1 sudo ./scripts/setup-environment.sh
  TORCH_CUDA_INDEX=https://download.pytorch.org/whl/cu124 sudo ./scripts/setup-environment.sh

EOF
}

main() {
    require_root
    resolve_target_user

    log "Проект: ${ROOT_DIR}"

    install_base_packages
    install_python_runtime
    resolve_python
    pip_install_args
    install_python_dependencies
    prepare_env_file
    verify_setup
    print_summary
}

main "$@"
