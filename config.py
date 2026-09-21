import anthropic
from dotenv import load_dotenv

OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "qwen3:30b"
ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
PREFILTER_THRESHOLD = 0.5
MAX_TO_SCORE = 20
MAX_TURNS = 10
OLLAMA_MAX_TOKENS = 100000

load_dotenv()

client = anthropic.Anthropic()
