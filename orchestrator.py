import os
import re
import json
import logging
import asyncio
from typing import Dict, Any
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
import tenacity
import time
from FigmaFrameAnalyzer import FigmaFrameAnalyzer
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

llm = ChatGroq(model_name="llama3-70b-8192", temperature=0, api_key=os.getenv("GROQ_API_KEY"))

@tenacity.retry(
    stop=tenacity.stop_after_attempt(5),
    wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
    retry=tenacity.retry_if_exception_type(Exception),
    before_sleep=lambda retry_state: logger.warning(f"Retrying after {retry_state.attempt_number} attempts...")
)
async def call_llm(prompt: PromptTemplate, input_data: Dict[str, Any]) -> str:
    try:
        chain = prompt | llm
        response = await chain.ainvoke(input_data)
        raw_text = response.content if hasattr(response, "content") else str(response)
        logger.info(f"LLM Response received for frame {input_data.get('frame_id')}")
        return raw_text
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        raise

async def process_single_frame(frame_id: str, frame_data: Dict[str, Any], project_description: str) -> Dict[str, Any]:
    try:
        analysis = frame_data.get("analysis", "")
        metadata = frame_data.get("metadata", "")
        error = frame_data.get("error", "")
        node_images = [url for url in frame_data.get("node_images", {}).values()]

        frame_summary = f"Frame ID: {frame_id}\n"
        if analysis:
            frame_summary += f"Analysis: {analysis}\n"
       

        prompt_template = f"""
        You are an AI tasked with analyzing a Figma frame to generate development tasks for a mobile app. Based on the frame summary and project description, provide tasks, a description, and navigation details.
        Project Description:
        {project_description}
        Frame Summary:
        {frame_summary}
        Instructions:
        Generate a JSON object with:
        "tasks" : List of task strings (e.g.,["Implement UI","Add navigation"]).The description should be elaborate and clear that a developer should know while coding. 
        "styling": List of styling strings (e.g.,["Use primary color for buttons","Apply shadow to cards"]).The description should be elaborate and clear that a developer should know while coding.
        "description": Brief description of the frame's purpose (max 50 words).
        "navigation": String describing navigation targets (e.g., "Navigates to frame X") or "None" if none.
        "node_images": List of image URLs used in the frame
        Focus on UI, interactions, and logic specific to this frame.
        
        Note : Output should only be in json format no other formats or words are entertained strictly.
        """
        prompt = PromptTemplate(
            input_variables=["project_description", "frame_summary","format_sample"],
            template=prompt_template
        )

        input_data = {
            "project_description": project_description,
            "frame_summary": frame_summary,
            "frame_id": frame_id  # For logging purposes
        }
        raw_text = await call_llm(prompt, input_data)

        try:
            result = json.loads(raw_text)
            return {
                "frame_id": frame_id,
                "tasks": result.get("tasks", []),
                "styling": result.get("styling", []),
                "description": result.get("description", ""),
                "navigation": result.get("navigation", "None"),
                "node_images": node_images,
                "metadata": metadata
            }
        
        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON for frame {frame_id}: {raw_text}")
            return {
                "frame_id": frame_id,
                "tasks": [],
                "styling": [],
                "description": "Failed to generate description",
                "navigation": "None",
                "node_images": node_images,
                "metadata": metadata,
                "error": "Failed to parse task JSON"
            }
    except Exception as e:
        logger.error(f"Failed to process frame {frame_id}: {e}")
        return {
            "frame_id": frame_id,
            "tasks": [],
            "description": "Processing failed",
            "navigation": "None",
            "node_images": node_images,
            "metadata": metadata,
            "error": str(e)
        }

async def extract_frames(figma_data: Dict[str, Any], project_description: str) -> Dict[str, Any]:
    try:
        frames = figma_data["frames"]
        frame_results = []

        for frame_id, frame_data in frames.items():
            logger.info(f"Processing frame: {frame_id}")
            result = await process_single_frame(frame_id, frame_data, project_description)
            frame_results.append(result)
            await asyncio.sleep(5)  

        output = {
            "project_name": figma_data.get("project_name", "figma_analysis"),
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "frames": frame_results
        }

        return output

    except Exception as e:
        logger.error(f"Frame extraction failed: {e}")
        raise

def analyze_codebase(base_path):
    issues = []
    for root, _, files in os.walk(base_path):
        for file in files:
            if file.endswith(".js") or file.endswith(".ts") or file.endswith(".tsx") or file.endswith(".jsx") or file.endswith(".py"):
                filepath = os.path.join(root, file)
                with open(filepath, "r", encoding="utf-8") as f:
                    code = f.read()

                if "console.log" in code or "print(" in code:
                    issues.append({
                        "file": filepath,
                        "issue": "Debug statement found"
                    })

                if re.search(r"\b(eval|exec)\b", code):
                    issues.append({
                        "file": filepath,
                        "issue": "Use of insecure function"
                    })

                if len(code.splitlines()) > 500:
                    issues.append({
                        "file": filepath,
                        "issue": "File too long"
                    })

    return issues

async def plan_tasks(project_description : str, figma_access_token : str, file_key : str) -> None:   
    analyzer = FigmaFrameAnalyzer(figma_access_token=figma_access_token)
    figma_data = analyzer.analyze_frames(file_key=file_key)
    tasks = await extract_frames(figma_data, project_description)
    return tasks

if __name__ == "__main__":
    asyncio.run(plan_tasks())