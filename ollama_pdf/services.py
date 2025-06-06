import logging
from pathlib import Path
from abc import ABC, abstractmethod
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents.stuff import (
    create_stuff_documents_chain,
)
from pydantic import SecretStr

logger = logging.getLogger(__name__)


class BaseLLMService(ABC):

    def serialize_pdf(self, file_path: Path) -> str:
        """LLM extracts relevant information

        Args:
            file_path (Path): Path to PDF file

        Returns:
            str: JSON in the defined format
        """
        docs = self._read_pdf(file_path)
        llm = self._get_llm_chain()
        return llm.invoke({"context": docs})

    @abstractmethod
    def _get_llm_chain(self) -> Runnable:
        """Get initiate LLM Chain

        Returns:
            Runnable: Runnable object
        """
        pass

    def _read_pdf(self, file_path: Path) -> list[Document]:
        """Reads the PDF file to LLM format

        Args:
            file_path (Path): Path to PDF file

        Returns:
            list[Document]: Pages of PDF
        """
        logger.info(f"read {file_path}")

        loader = PyPDFLoader(file_path)
        return loader.load()

    def _create_prompt(self, prompt_base: str) -> PromptTemplate:
        """Creates a PromptTemplate object for the LLM Chain

        Args:
            prompt_base (str): Prompt to define what and
                               how to extract structured data.

        Returns:
            PromptTemplate: PromptTemplate object
        """
        # escape curly brackets to comply to langchain
        prompt_base = prompt_base.replace("{", "{{").replace("}", "}}")
        # add placeholder for the document
        prompt_suffix = '\nDocument: "{context}" JSON:'
        prompt = prompt_base + prompt_suffix

        return PromptTemplate.from_template(prompt)

    def _create_chain(self, llm: BaseChatModel, prompt_text: str) -> Runnable:
        """_summary_

        Args:
            llm (BaseChatModel): LLM Chat Model
            prompt_text (str): Prompt to define what and
                               how to extract structured data

        Returns:
            Runnable: Runnable object
        """

        # bind the output to json format
        llm_json = llm.bind(response_format={"type": "json_object"})

        return create_stuff_documents_chain(
            llm=llm_json, prompt=self._create_prompt(prompt_text)
        )


class OpenAIService(BaseLLMService):
    """Service to call OpenAI capable LLM servers"""

    def __init__(
        self,
        base_url: str,
        api_key: SecretStr,
        model: str,
        prompt_text: str,
        **kwargs,
    ):
        """Configure OpenAI LLM server to revice a PDF and
        return extracted data in JSON

        Args:
            base_url (str): base url of the server
                (i.e. http://localhost:11434/v1)
            api_key (str): api_key to identify
            model (str): LLM model to use
            prompt_text (str): Prompt to define what and
                               how to extract structured data
        """

        llm = ChatOpenAI(
            temperature=0,
            model=model,
            api_key=api_key,
            base_url=base_url,
        )

        self._llm_chain = self._create_chain(llm, prompt_text)

    def _get_llm_chain(self) -> Runnable:
        """Get initiate LLM Chain

        Returns:
            Runnable: Runnable object
        """
        return self._llm_chain


class LLMServiceFactory:
    """Factory to create LLM Service"""

    @classmethod
    def createOpenAI(
        cls,
        base_url: str,
        api_key: SecretStr,
        model: str,
        prompt: str,
        **kwargs,
    ) -> BaseLLMService:
        """creates an OpenAIService

        Args:
            base_url (str): base url of the server
                (i.e. http://localhost:11434/v1)
            api_key (str): api_key to identify
            model (str): LLM model to use
            prompt (str): Prompt to define what and
                          how to extract structured data

        Returns:
            OpenAIService: The OpenAiService object
        """
        logger.info("create OpenAI LLM Server")
        return OpenAIService(
            base_url=base_url, api_key=api_key, model=model, prompt_text=prompt
        )
