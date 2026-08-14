"""Load .env before collection, so test_router.py's skipif sees OPENROUTER_API_KEY
regardless of which test file pytest is invoked with."""

from dotenv import load_dotenv

load_dotenv()
