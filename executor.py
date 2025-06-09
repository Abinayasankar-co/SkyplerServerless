import os
import re
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from tools.file_writer import buffered_write
from dotenv import load_dotenv


load_dotenv()

llm = ChatGroq(model_name="llama3-70b-8192", temperature=0 , api_key=os.getenv("GROQ_API_KEY"))

def extract_code(content: str) -> str:
    match = re.search(r"```(?:\w*\n)?(.*?)```", content, re.DOTALL)
    return match.group(1).strip() if match else content.strip()

def execute_task(task):
    prompt = PromptTemplate.from_file("prompts/codegen_prompt.txt", input_variables=[
        "project_type", "module", "file_type", "file_path", "description"
    ])
    chain = prompt | llm
    response = chain.invoke(task)
    raw_text = response.content if hasattr(response, "content") else str(response)
    code = extract_code(raw_text)
    full_path = f'generated_app/{task["file_path"]}'
    buffered_write(full_path, code)
    return True
