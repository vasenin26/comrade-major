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
2. Установку системных пакетов через `apt` (Python 3.12, PortAudio, ffmpeg и др.)
3. `pip install -e .[dev]` в **системный** Python (с `--break-system-packages` на Debian/Ubuntu с PEP 668)
4. Создание `.env` из `.env.example`, если `.env` ещё нет
5. Быструю проверку импортов и `pytest`

### Ручная установка

```bash
# Системные пакеты (Ubuntu/Debian/WSL)
sudo apt update
sudo apt install -y ca-certificates curl git build-essential pkg-config \
    portaudio19-dev libportaudio2 libsndfile1 ffmpeg \
    python3.12 python3.12-dev python3-pip

# Python-зависимости в system Python
cd comrade-major
python3.12 -m pip install --upgrade pip --break-system-packages
python3.12 -m pip install -e ".[dev]" --break-system-packages

# Конфигурация
cp .env.example .env
```

> Флаг `--break-system-packages` нужен на современных Debian/Ubuntu из‑за PEP 668. На выделенной машине под агента это ожидаемое поведение.

### GPU (опционально)

Для ускорения LLM/STT на NVIDIA GPU установите [драйвер CUDA](https://developer.nvidia.com/cuda-downloads) и переустановите PyTorch:

```bash
python3.12 -m pip install torch torchaudio \
    --index-url https://download.pytorch.org/whl/cu124 \
    --break-system-packages
```

В `.env` оставьте `LLM_DEVICE=auto`.

## Быстрый старт

```bash
cp .env.example .env   # если ещё не создан
python3.12 -m src.main
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
python3.12 -m pytest
python3.12 -m mypy config src
```
