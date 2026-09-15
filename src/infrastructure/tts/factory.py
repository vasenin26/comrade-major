from config.settings import Settings
from src.application.interfaces import TTSService
from src.infrastructure.tts.silero import SileroTTS


def create_tts(settings: Settings) -> TTSService:
    return SileroTTS(
        sample_rate=settings.sample_rate,
        speaker=settings.tts_speaker,
        device=settings.resolved_tts_device(),
        synthesis_sample_rate=settings.tts_synthesis_sample_rate,
    )
