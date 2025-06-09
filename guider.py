from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from tools.file_writer import overwrite_file
from tools.error_parser import extract_file_error
import subprocess
import os
import re
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model_name="llama3-70b-8192", temperature=0, api_key=os.getenv("GROQ_API_KEY"))

def validate_codebase():
    try:
        result = subprocess.run(
            ["npx", "eslint", "."],
            cwd="generated_app",
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode != 0:
            return result.stderr
    except Exception as e:
        return str(e)
    return None

def run_guider(state):
    while True:
        error = validate_codebase()
        if not error:
            break
        affected = extract_file_error(error)
        for item in affected:
            patch_code(state["project_type"], item["file_path"], item["message"])
    return state

def patch_code(project_type, file_path, message):
    prompt = PromptTemplate.from_file("prompts/fixer_prompt.txt", input_variables=[
        "project_type", "file_path", "message", "original_code"
    ])
    with open(f"generated_app/{file_path}", "r", encoding="utf-8") as f:
        original_code = f.read()
    chain = prompt | llm
    response = chain.invoke({
        "project_type": project_type,
        "file_path": file_path,
        "message": message,
        "original_code": original_code
    })
    code = extract_code_block(response.content if hasattr(response, "content") else str(response))
    overwrite_file(f"generated_app/{file_path}", code)

def extract_code_block(text):
    match = re.search(r"```(?:\w*\n)?(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()
