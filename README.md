# Voice Agent

Асинхронный голосовой ИИ-агент со стримингом аудио и поддержкой barge-in (прерывание TTS речью пользователя).

Проект рассчитан на **выделенную машину**: Python и зависимости ставятся **системно**, без виртуальных окружений.

## Стек

- Python **3.12+** (системный)
- pip — установка зависимостей в system Python
- HuggingFace transformers, faster-whisper, Silero VAD, sounddevice
- LLM: локально (`local`) или внешний OpenAI-compatible API (`openai`)

## Структура

```
comrade-major/
├── config/                 # Pydantic-настройки
├── scripts/
│   └── setup-environment.sh
├── src/
│   ├── main.py             # Точка входа, сборка DI
│   ├── domain/             # Бизнес-логика (чат)
│   ├── application/        # Координатор, Protocol-интерфейсы
│   └── infrastructure/     # Аудио, VAD, STT, LLM, TTS
├── tests/
├── pyproject.toml
├── .env.example
└── .cursor/rules/          # Правила для Cursor AI
```

## Подготовка среды

### Требования

| Компонент | Назначение |
|-----------|------------|
| Python 3.12+ | Системный runtime |
| pip | Установка Python-зависимостей |
| PortAudio (`portaudio19-dev`) | Захват/воспроизведение звука (`sounddevice`) |
| libsndfile | Чтение аудиофайлов (`soundfile`) |
| ffmpeg | Обработка аудио |
| build-essential | Сборка нативных расширений |
| git | Загрузка моделей (Silero VAD через torch.hub) |

Поддерживается **Linux (Debian/Ubuntu/WSL)**. На WSL2 для микрофона может понадобиться включить аудио в настройках Windows/WSL.

### Автоматическая установка (рекомендуется)

```bash
chmod +x scripts/setup-environment.sh
sudo ./scripts/setup-environment.sh
```

Скрипт выполняет:

1. Проверку прав root (`sudo`)
2. Установку системных пакетов через `apt` (PortAudio, ffmpeg и др.)
3. Автовыбор **Python 3.12+** из apt (предпочитает 3.12; на Debian 13 без deadsnakes — 3.13)
4. **Проверку NVIDIA/CUDA** (`nvidia-smi`) и установку PyTorch с подходящим index (cu126/cu124/cu118 или CPU)
5. `pip install -e .[dev]` в **системный** Python
6. Smoke-test CUDA в PyTorch и `pytest`

> Скрипт сам определяет версию Python (3.12, 3.13 или 3.14) — не нужно вручную ставить именно 3.12.

### Ручная установка

```bash
# Системные пакеты (Ubuntu/Debian/WSL)
sudo apt update
sudo apt install -y ca-certificates curl git build-essential pkg-config \
    portaudio19-dev libportaudio2 libsndfile1 ffmpeg \
    python3.13 python3.13-dev python3-pip   # или python3.12 на Ubuntu 24.04

# Python-зависимости (сначала PyTorch — CPU или CUDA)
cd comrade-major
python3.12 -m pip install --upgrade pip --break-system-packages
python3.12 -m pip install torch torchaudio \
    --index-url https://download.pytorch.org/whl/cu124 \
    --break-system-packages
python3.12 -m pip install -e ".[dev]" --break-system-packages

# Конфигурация
cp .env.example .env
```

> Флаг `--break-system-packages` нужен на Debian/Ubuntu с PEP 668.

### GPU / CUDA

Скрипт `setup-environment.sh` **сам определяет GPU**:

1. Запускает `nvidia-smi`
2. Читает версию CUDA из драйвера
3. Ставит PyTorch с нужным index:
   - CUDA 12.6+ / 13.x → `cu126`
   - CUDA 12.4–12.5 → `cu124`
   - CUDA 11.x → `cu118`
   - GPU нет → CPU index
4. Проверяет `torch.cuda.is_available()` и делает smoke-test на GPU

Если `nvidia-smi` видит GPU, но PyTorch CUDA недоступна — скрипт завершится с ошибкой.

Переопределение:

```bash
FORCE_CPU=1 sudo ./scripts/setup-environment.sh
TORCH_CUDA_INDEX=https://download.pytorch.org/whl/cu124 sudo ./scripts/setup-environment.sh
```

В `.env` оставьте `LLM_DEVICE=auto`.

## Быстрый старт

```bash
cp .env.example .env   # если ещё не создан
python3.13 -m src.main   # или python3.12 — та версия, что установилась
```

При первом запуске модели скачаются с HuggingFace (локальный LLM, Whisper, Silero VAD).

## Конфигурация

Параметры в `config/settings.py`, переопределение через `.env`. Шаблон — `.env.example`:

```env
SAMPLE_RATE=16000
VAD_THRESHOLD=0.5
WHISPER_MODEL_SIZE=small

LLM_PROVIDER=local
LLM_MODEL=Qwen/Qwen2.5-0.5B-Instruct
LLM_DEVICE=auto
LLM_MAX_NEW_TOKENS=256
LLM_TEMPERATURE=0.7
```

Для внешнего LLM-провайдера:

```env
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
LLM_API_KEY=sk-...
LLM_BASE_URL=https://api.openai.com/v1
```

## Архитектурные принципы

1. **Non-blocking asyncio** — инференс и I/O через `asyncio.to_thread`.
2. **In-memory пайплайн** — аудио как `numpy` массивы между модулями.
3. **Слоистая архитектура** — Domain / Application / Infrastructure.
4. **Ручной DI** — сборка зависимостей в `main.py`.

## Тесты и типы

```bash
python3.13 -m pytest
python3.13 -m mypy config src
```
