import logging
import sys
from pathlib import Path
from datetime import datetime
from ollama_pdf.services import LLMServiceFactory
from ollama_pdf.utils import EnvConfigReader
from pprint import pprint

format = "[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s"
logging.basicConfig(
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
    format=format,
)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    # init paths
    today = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    input_path = Path("./data")
    output_path = Path("./results") / today
    models = [
        "phi3:latest",
        "llama3.2:latest",
        "gemma3:4b",
        "Qwen3:4b",

    ]

    # create folders if necessary
    input_path.mkdir(parents=True, exist_ok=True)
    output_path.mkdir(parents=True, exist_ok=True)

    # read config
    logger.info("read config")
    config = EnvConfigReader()

    for model in models:
        file_name = model.replace(":", "_")
        logger.info(f"Processing model: {model}")
        out_file = output_path / f"{file_name}.txt"
        llm_service = LLMServiceFactory.create_service(
            config.service,
            config.base_url,
            config.api_key,
            model,
            config.embedding_model,
            config.prompt,
        )

        # get files
        with open(out_file, "a") as result_file:
            for file_path in input_path.glob("*.pdf"):
                response = llm_service.extract_structure(file_path)
                logger.info(f"File {file_path} done")
                result_file.write(f"{file_path}:\n")
                result_file.write(str(response["result"].model_dump()))
                result_file.write("\n\n")
                pprint(response["result"].model_dump())
