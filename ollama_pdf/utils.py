import os
import logging
from abc import ABC
from pathlib import Path
from dotenv import load_dotenv
from pydantic import SecretStr


logger = logging.getLogger(__name__)


class BaseConfigReader(ABC):
    """Base class for config readers"""

    def __init__(self):
        """Reads config from environment variables"""
        self._config = {
            "OPDF_SERVICE": "OpenAI",
            "OPDF_BASE_URL": "https://api.openai.com/v1",
            "OPDF_MODEL": "gpt-3.5-turbo",
            "OPDF_EMBEDDING_MODEL": "text-embedding-ada-002",
            "OPDF_API_KEY": "your_api_key_here",
            "OPDF_PROMPT": "Please summarize the following PDF content.",
        }

    def _load_prompt_file(self, prompt_path: Path):
        """Loads the prompt from a file specified in the config.

        Args:
            prompt_path (Path): Path to the prompt file.
        """

        if prompt_path.exists():
            with open(prompt_path, "r") as prompt_file:
                self._config["OPDF_PROMPT"] = prompt_file.read()

    @property
    def service(self) -> str:
        """Returns the LLM Service to use"""
        return self._config["OPDF_SERVICE"]

    @property
    def base_url(self) -> str:
        """Returns the base URL to use for the LLM Service"""
        return self._config["OPDF_BASE_URL"]

    @property
    def model(self) -> str:
        """Returns the model to use for the LLM Service"""
        return self._config["OPDF_MODEL"]

    @property
    def embedding_model(self) -> str:
        """Returns the embedding model to use for the LLM Service"""
        return self._config["OPDF_EMBEDDING_MODEL"]

    @property
    def api_key(self) -> SecretStr:
        """Returns the API key to use for the LLM Service"""
        return SecretStr(self._config["OPDF_API_KEY"])
        

    @property
    def prompt(self) -> str:
        """Returns the prompt to use for the LLM"""
        return self._config["OPDF_PROMPT"]


class EnvConfigReader(BaseConfigReader):
    """Object to read an store configs from environment variables"""

    def __init__(self):
        """Reads config from environment variables"""
        super().__init__()
        load_dotenv()
        self._load_env_vars()

    def _load_env_vars(self):
        """Loads config from environment variables."""

        self._config = {
            k: v for k, v in os.environ.items() if k.startswith("OPDF")
        }
        if "OPDF_PROMPT_FILE" in self._config:
            prompt_path = Path(self._config.get("OPDF_PROMPT_FILE", ""))
            self._load_prompt_file(prompt_path)
