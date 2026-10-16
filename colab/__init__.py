"""The Colab VM side of the bot's Colab tier (ADR-0010, colab/README.md).

The bot's launcher uploads ``VM_FILES`` flat into ``/content`` on a freshly
assigned Colab runtime and starts ``bootstrap.py`` there as a detached
process. On the VM the modules import each other as plain siblings; in this
repository they form the ``colab`` package so the bot and the tests can
import them. They use the standard library only (``auth_proxy`` imports
aiohttp lazily, when it serves), so importing them costs the bot nothing.
"""

from pathlib import Path

# The pins and the Ollama fallback; the launcher uploads a copy whose
# "ollama" entry is the release it picked for the session.
VERSIONS_FILE = Path(__file__).resolve().with_name("versions.json")

VM_FILES: tuple[str, ...] = (
    "bootstrap.py",
    "auth_proxy.py",
    "register.py",
    "versions.json",
    "proxy-requirements.lock",
)
