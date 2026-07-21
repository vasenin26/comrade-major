#!/usr/bin/env bash
# Подготовка системного окружения для voice-agent.
# Python-пакеты ставятся в систему (без venv) — проект рассчитан на выделенную машину.
# Запуск: sudo ./scripts/setup-environment.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MIN_PYTHON_VERSION="3.12"

APT_PACKAGES=(
    ca-certificates
    curl
    git
    build-essential
    pkg-config
    portaudio19-dev
    libportaudio2
    libsndfile1
    ffmpeg
    python3.12
    python3.12-dev
    python3-pip
)

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

install_apt_packages() {
    if ! have_command apt-get; then
        die "apt-get не найден. Скрипт рассчитан на Debian/Ubuntu/WSL."
    fi

    log "Обновляю индекс пакетов apt..."
    apt-get update -qq

    local missing=()
    for package in "${APT_PACKAGES[@]}"; do
        if ! dpkg -s "${package}" >/dev/null 2>&1; then
            missing+=("${package}")
        fi
    done

    if ((${#missing[@]} == 0)); then
        log "Все системные пакеты уже установлены."
        return 0
    fi

    log "Устанавливаю системные пакеты: ${missing[*]}"
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "${missing[@]}"
}

resolve_python() {
    if have_command python3.12; then
        PYTHON_BIN="$(command -v python3.12)"
    elif have_command python3; then
        PYTHON_BIN="$(command -v python3)"
    else
        die "Python не найден. Установите python3.12 через apt."
    fi

    log "Использую Python: ${PYTHON_BIN}"
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

pip_install_args() {
    PIP_ARGS=()
    if [[ -f /usr/lib/python3.12/EXTERNALLY-MANAGED ]] || [[ -f /usr/lib/python3/EXTERNALLY-MANAGED ]]; then
        PIP_ARGS+=(--break-system-packages)
        log "PEP 668: использую --break-system-packages для системной установки."
    fi
}

install_python_dependencies() {
    log "Обновляю pip..."
    "${PYTHON_BIN}" -m pip install --upgrade pip "${PIP_ARGS[@]:-}"

    log "Устанавливаю voice-agent и dev-зависимости в системный Python..."
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

verify_setup() {
    log "Проверяю установку..."
    "${PYTHON_BIN}" -c 'import sounddevice, torch, transformers; print("imports: OK")'
    (cd "${ROOT_DIR}" && "${PYTHON_BIN}" -m pytest -q)
}

print_summary() {
    cat <<EOF

Готово. Зависимости установлены в системный Python (${PYTHON_BIN}).

Дальнейшие шаги:
  cd ${ROOT_DIR}
  cp .env.example .env   # если .env ещё не создан
  ${PYTHON_BIN} -m src.main

Проверки:
  ${PYTHON_BIN} -m pytest
  ${PYTHON_BIN} -m mypy config src

EOF
}

main() {
    require_root
    resolve_target_user

    log "Проект: ${ROOT_DIR}"

    install_apt_packages
    resolve_python
    pip_install_args
    install_python_dependencies
    prepare_env_file
    verify_setup
    print_summary
}

main "$@"
