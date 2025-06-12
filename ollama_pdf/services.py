import logging
from pathlib import Path
from abc import ABC, abstractmethod
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import Runnable
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import SecretStr
from .model import FinalResponse


logger = logging.getLogger(__name__)


class BaseLLMService(ABC):

    def __init__(self, prompt_text: str):
        """Configure LLM server to revice a PDF and
        return extracted data in JSON

        Args:
            prompt_text (str): Prompt to define what and
                               how to extract structured data
        """
        self._prompt_text = prompt_text

    def extract_structure(self, file_path: Path) -> dict:
        """LLM extracts relevant information

        Args:
            file_path (Path): Path to PDF file
        Returns:
            dict: Response from LLM with extracted data
        """
        docs = self._read_pdf(file_path)

        llm = self._get_llm()
        vector_store = self._create_vector_store(docs)
        llm_chain = self._create_chain(llm, vector_store.as_retriever())
        repsonse: dict = llm_chain.invoke(self._prompt_text)
        vector_store.delete_collection()
        return repsonse

    def _read_pdf(self, file_path: Path) -> list[Document]:
        """Reads the PDF file to LLM format and splits into chunks

        Args:
            file_path (Path): Path to PDF file

        Returns:
            list[Document]: Pages of PDF
        """
        logger.info(f"read {file_path}")

        loader = PyPDFLoader(file_path)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=100
        )

        return loader.load_and_split(text_splitter=text_splitter)

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

    def _create_vector_store(self, docs: list[Document]) -> Chroma:
        """Creates a vector store retriever from the documents
        Args:
            docs (list[str]): List of documents to be stored
        Returns:
            VectorStoreRetriever: Vector store retriever
        """

        vectordb = Chroma.from_documents(
            documents=docs,
            embedding=self._get_embeddings(),
        )
        return vectordb

    def _create_chain(
        self, llm: BaseChatModel, vector_store: VectorStoreRetriever
    ) -> Runnable:
        """_summary_

        Args:
            llm (BaseChatModel): LLM Chat Model
            prompt_text (str): Prompt to define what and
                               how to extract structured data

        Returns:
            Runnable: Runnable object
        """

        # bind the output to json format
        # llm_json = llm.bind(response_format={"type": "json_object"})
        llm_json = llm.with_structured_output(FinalResponse)
        return RetrievalQA.from_chain_type(
            llm=llm_json,
            chain_type="stuff",
            retriever=vector_store,
            verbose=True,
        )

    @abstractmethod
    def _get_llm(self) -> BaseChatModel:
        """Get LLM Chain

        Returns:
            Runnable: Runnable object
        """
        pass

    @abstractmethod
    def _get_embeddings(self) -> Embeddings:
        """Get Embedding specific to the LLM service
        This method initializes the Embedding object

        Returns:
            Embeddings: Embedding
        """
        pass


class OpenAIService(BaseLLMService):
    """Service to call OpenAI capable LLM servers"""

    def __init__(
        self,
        base_url: str,
        api_key: SecretStr,
        model: str,
        embedding_model: str,
        prompt_text: str,
    ):
        """Configure OpenAI LLM server to revice a PDF and
        return extracted data in JSON

        Args:
            base_url (str): base url of the server
                (i.e. http://localhost:11434/v1)
            api_key (str): api_key to identify
            model (str): LLM model to use
            embedding_model (str): Embedding model to use
            prompt_text (str): Prompt to define what and
                how to extract structured data
            vector_store_path (Path, optional): Path to vector store.
                Defaults to Path("vector_store").
        """
        super().__init__(prompt_text)
        self._model: str = model
        self._embedding_model: str = embedding_model
        self._base_url: str = base_url
        self._api_key: SecretStr = api_key
        self._llm: BaseChatModel | None = None
        self._embeddings: Embeddings | None = None

    def _get_llm(self) -> BaseChatModel:
        """Get initiate LLM Chain

        Returns:
            Runnable: Runnable object
        """
        if not self._llm:
            self._llm = ChatOpenAI(
                temperature=0,
                model=self._model,
                api_key=self._api_key,
                base_url=self._base_url,
            )

        return self._llm

    def _get_embeddings(self) -> Embeddings:
        """Get Embedding specific to OpenAI
        This method initializes the OpenAIEmbeddings object

        Returns:
            Embeddings: Embedding
        """
        if not self._embeddings:
            self._embeddings = OpenAIEmbeddings(
                model=self._embedding_model,
                base_url=self._base_url,
                api_key=self._api_key,
            )
        return self._embeddings


class OllamaService(BaseLLMService):
    """Service to call OpenAI capable LLM servers"""

    def __init__(
        self,
        base_url: str,
        api_key: SecretStr,
        model: str,
        embedding_model: str,
        prompt_text: str,
    ):
        """Configure OpenAI LLM server to revice a PDF and
        return extracted data in JSON

        Args:
            base_url (str): base url of the server
                (i.e. http://localhost:11434)
            api_key (str): api_key to identify
            model (str): LLM model to use
            embedding_model (str): Embedding model to use
            prompt_text (str): Prompt to define what and
                how to extract structured data
            vector_store_path (Path, optional): Path to vector store.
                Defaults to Path("vector_store").
        """
        super().__init__(prompt_text)
        self._model: str = model
        self._embbedding_model: str = embedding_model
        self._base_url: str = base_url
        self._api_key: SecretStr = api_key
        self._llm: BaseChatModel | None = None
        self._embeddings: Embeddings | None = None

    def _get_client_kwargs(self) -> dict | None:
        """Get sync kwargs for the LLM and Embeddings

        Returns:
            dict | None: Sync kwargs for the LLM and Embeddings
        """
        if not self._api_key:
            return {}

        return {
            "headers": {
                "Authorization": f"Bearer {self._api_key.get_secret_value()}"
            }
        }

    def _get_llm(self) -> BaseChatModel:
        """Get initiate LLM Chain

        Returns:
            Runnable: Runnable object
        """
        client_kwargs = self._get_client_kwargs()
        if not self._llm:
            self._llm = ChatOllama(
                temperature=0,
                model=self._model,
                base_url=self._base_url,
                client_kwargs=client_kwargs,
            )
        return self._llm

    def _get_embeddings(self) -> Embeddings:
        """Get Embedding specific to OpenAI
        This method initializes the OpenAIEmbeddings object

        Returns:
            Embeddings: Embedding
        """
        if not self._embeddings:
            client_kwargs = self._get_client_kwargs()
            self._embeddings = OllamaEmbeddings(
                model=self._embbedding_model,
                base_url=self._base_url,
                sync_client_kwargs=client_kwargs,
            )
        return self._embeddings


class LLMServiceFactory:
    """Factory to create LLM Service"""

    @classmethod
    def create_service(
        cls,
        service: str,
        base_url: str,
        api_key: SecretStr | None = None,
        model: str = "gpt-3.5-turbo",
        embedding_model: str = "text-embedding-ada-002",
        prompt: str = "Extract the structured data from the document.",
    ) -> BaseLLMService:
        """Creates a LLM Service based on the service type

        Args:
            service (str): Type of LLM Service to create
            base_url (str): base url of the server
                (i.e. http://localhost:11434/v1)
            api_key (SecretStr, optional): API key to identify.
                Defaults to None.
            model (str, optional): LLM model to use.
                Defaults to "gpt-3.5-turbo".
            embedding_model (str, optional): Embedding model to use.
                Defaults to "text-embedding-ada-002".
            prompt (str, optional): Prompt to define what and
                how to extract structured data.
                Defaults to "Extract the structured data from the document.".

        Returns:
            BaseLLMService: The created LLM Service object
        """
        if service == "OpenAI":
            if api_key is None:
                raise ValueError("API key is required for OpenAI service.")
            return cls._create_OpenAI(
                base_url, api_key, model, embedding_model, prompt
            )
        elif service == "Ollama":
            if api_key is None:
                raise ValueError("API key is required for Ollama service.")
            return cls._create_Ollama(
                base_url, model, api_key, embedding_model, prompt
            )
        else:
            raise ValueError(f"Unknown service type: {service}")

    @classmethod
    def _create_OpenAI(
        cls,
        base_url: str,
        api_key: SecretStr,
        model: str,
        embedding_model: str,
        prompt: str,
    ) -> BaseLLMService:
        """creates an OpenAIService

        Args:
            base_url (str): base url of the server
                (i.e. http://localhost:11434/v1)
            api_key (str): api_key to identify
            model (str): LLM model to use
            embedding_model (str): Embedding model to use
            prompt (str): Prompt to define what and
                          how to extract structured data

        Returns:
            OpenAIService: The OpenAiService object
        """
        logger.info("create OpenAI LLM Server")
        return OpenAIService(
            base_url=base_url,
            api_key=api_key,
            model=model,
            embedding_model=embedding_model,
            prompt_text=prompt,
        )

    @classmethod
    def _create_Ollama(
        cls,
        base_url: str,
        model: str,
        api_key: SecretStr,
        embedding_model: str,
        prompt: str,
    ) -> BaseLLMService:
        """creates an OllamaService

        Args:
            base_url (str): base url of the server
                (i.e. http://localhost:11434/v1)
            model (str): LLM model to use
            embedding_model (str): Embedding model to use
            prompt (str): Prompt to define what and
                          how to extract structured data

        Returns:
            OpenAIService: The OpenAiService object
        """
        logger.info("create Ollama LLM Server")
        return OllamaService(
            base_url=base_url,
            model=model,
            api_key=api_key,
            embedding_model=embedding_model,
            prompt_text=prompt,
        )
