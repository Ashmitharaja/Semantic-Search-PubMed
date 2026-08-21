"""
config.py
==========
NCBI credentials, read from environment variables (or a local .env file),
NOT exposed as UI inputs. Set these once here on the machine running the
app, instead of typing them into the browser every time.

  NCBI_EMAIL    - your contact email. Recommended by NCBI, not required.
  NCBI_API_KEY  - your NCBI API key. Raises the rate limit from 3 req/sec
                  (anonymous) to 10 req/sec. Get one free at:
                  https://www.ncbi.nlm.nih.gov/account/settings/

How to set these
-----------------
Option A - environment variables (any OS):
    macOS/Linux:
        export NCBI_EMAIL="you@example.com"
        export NCBI_API_KEY="your_key_here"
    Windows (PowerShell):
        $env:NCBI_EMAIL="you@example.com"
        $env:NCBI_API_KEY="your_key_here"

Option B - a local .env file (see .env.example in this folder):
    1. Copy .env.example to .env
    2. Fill in your real values
    3. Install python-dotenv (already in requirements.txt) - this module
       loads .env automatically on import, no extra code needed.

Either the app or eval_harness.py will pick these up automatically via
`import config`; nothing else in the codebase needs to change when you set
real credentials.

If NCBI_API_KEY is left unset, the app still works correctly end to end -
ncbi_client._throttle() automatically falls back to the slower (but fully
functional) 3 req/sec anonymous rate limit that NCBI enforces for
unauthenticated requests. Nothing breaks; searches just self-limit to a
slower pace.
"""
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# First try Streamlit Cloud Secrets
try:
    import streamlit as st

    NCBI_EMAIL = st.secrets.get("NCBI_EMAIL", "")
    NCBI_API_KEY = st.secrets.get("NCBI_API_KEY", "")

except Exception:
    # Local environment / .env
    NCBI_EMAIL = os.environ.get("NCBI_EMAIL", "")
    NCBI_API_KEY = os.environ.get("NCBI_API_KEY", "")


if not NCBI_API_KEY:
    print(
        "[Bio-Lens] NCBI_API_KEY not set - running at NCBI's slower "
        "anonymous rate limit (3 req/sec) instead of 10 req/sec. See "
        "config.py for how to set it. Get a free key at "
        "https://www.ncbi.nlm.nih.gov/account/settings/"
    )

