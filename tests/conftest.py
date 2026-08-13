"""Ensures .env is loaded before any test module's collection-time code runs (e.g.
test_router.py's skipif check on OPENROUTER_API_KEY) - regardless of which specific test
file or subset pytest is invoked with. Previously this only worked by accident when the
full suite ran, because test_main.py's import of app.main happened to trigger load_dotenv()
as a side effect before test_router.py's skip condition was evaluated; running
`pytest tests/test_router.py` alone skipped every test since nothing had loaded .env yet.
"""

from dotenv import load_dotenv

load_dotenv()
