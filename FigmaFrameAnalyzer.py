import os
import base64
import json
import requests
import time
from groq import Groq
from typing import Dict, List, Optional
from dotenv import load_dotenv
from datetime import datetime
from PIL import Image
from io import BytesIO
import logging
from langgraph.graph import StateGraph, END
from langgraph.graph import StateGraph
from typing import TypedDict

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class State(TypedDict):
    figma_frames: List[Dict]
    image_nodes: Dict[str, Dict]
    image_urls: Dict[str, str]
    node_image_urls: Dict[str, str]
    frame_content: Dict[str, Dict]
    current_frame_index: int
    file_key: str
    figma_access_token: str

class FigmaFrameAnalyzer:
    def __init__(self, figma_access_token: str, model_vlm: str = "meta-llama/llama-4-scout-17b-16e-instruct"):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.figma_access_token = figma_access_token
        self.model_vlm = model_vlm
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(State)

        graph.add_node("validate_figma_access", self._validate_figma_access)
        graph.add_node("fetch_figma_frames", self._fetch_figma_frames)
        graph.add_node("fetch_image_urls", self._fetch_image_urls)
        graph.add_node("process_frame", self._process_frame)
        graph.add_node("verify_content", self._verify_content)
        graph.add_node("compile_results", self._compile_results)

        graph.add_edge("validate_figma_access", "fetch_figma_frames")
        graph.add_edge("fetch_figma_frames", "fetch_image_urls")
        graph.add_edge("fetch_image_urls", "process_frame")
        graph.add_conditional_edges(
            "process_frame",
            self._check_more_frames,
            {
                "next_frame": "process_frame",
                "done": "verify_content"
            }
        )
        graph.add_edge("verify_content", "compile_results")
        graph.add_edge("compile_results", END)

        graph.set_entry_point("validate_figma_access")
        return graph.compile()

    def _validate_figma_access(self, state: State) -> Dict:
        url = f"https://api.figma.com/v1/files/{state['file_key']}"
        headers = {"X-Figma-Token": state['figma_access_token']}
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            logger.info("Figma access validated successfully")
            return {}
        except requests.exceptions.HTTPError as e:
            logger.error(f"Figma API error: {str(e)}")
            return {"figma_frames": []}
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to validate Figma access: {str(e)}")
            return {"figma_frames": []}

    def _fetch_figma_frames(self, state: State) -> Dict:
        url = f"https://api.figma.com/v1/files/{state['file_key']}"
        headers = {"X-Figma-Token": state['figma_access_token']}
        figma_frames = []
        image_nodes = state['image_nodes']
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()
            logger.info("Figma API response received")
            document = data.get("document", {})
            children = document.get("children", [])
            logger.info(f"Found {len(children)} top-level nodes in Figma file")
            
            figma_frames = self._find_frames_recursive(children)
            logger.info("Detected frames: " + ", ".join([frame["name"] for frame in figma_frames if "name" in frame]))
            
            for frame in figma_frames:
                frame_id = frame.get("id")
                if "children" in frame:
                    image_nodes.update(self._find_image_nodes(frame["children"], frame_id))
            
            if not figma_frames:
                logger.warning("No FRAME nodes found. Checking for CANVAS nodes...")
                for node in children:
                    if node.get("type") == "CANVAS" and "children" in node:
                        figma_frames.extend(self._find_frames_recursive(node["children"]))
                        for frame in figma_frames:
                            frame_id = frame.get("id")
                            if "children" in frame:
                                image_nodes.update(self._find_image_nodes(frame["children"], frame_id))
                        logger.info("Detected frames after CANVAS: " + ", ".join([frame["name"] for frame in figma_frames if "name" in frame]))
            
            logger.info(f"Total frames found: {len(figma_frames)}")
            logger.info(f"Total image nodes found: {len(image_nodes)}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch Figma frames: {str(e)}")
            figma_frames = []
        return {"figma_frames": figma_frames, "image_nodes": image_nodes}

    def _find_frames_recursive(self, nodes: List[Dict], depth: int = 0, max_depth: int = 1) -> List[Dict]:
        frames = []
        for node in nodes:
            if node.get("type") == "FRAME" and depth == 0:
                frames.append(node)
            if depth < max_depth and "children" in node:
                frames.extend(self._find_frames_recursive(node["children"], depth + 1, max_depth))
        return frames

    def _find_image_nodes(self, nodes: List[Dict], frame_id: str = None) -> Dict[str, Dict]:
        image_nodes = {}
        for node in nodes:
            node_id = node.get("id")
            node_name = node.get("name", "unknown").replace(" ", "_")
            if node.get("type") == "IMAGE":
                image_nodes[node_id] = {"name": node_name, "frame_id": frame_id}
            elif "fills" in node and node["fills"]:
                for fill in node["fills"]:
                    if fill.get("type") == "IMAGE" and "imageRef" in fill:
                        image_nodes[node_id] = {"name": node_name, "frame_id": frame_id, "imageRef": fill["imageRef"]}
            if "children" in node:
                image_nodes.update(self._find_image_nodes(node["children"], frame_id))
        return image_nodes

    def _fetch_image_urls(self, state: State) -> Dict:
        image_urls = {}
        node_image_urls = {}
        frame_ids = [frame["id"] for frame in state['figma_frames']]
        if frame_ids:
            url = f"https://api.figma.com/v1/images/{state['file_key']}?ids={','.join(frame_ids)}&format=jpg"
            headers = {"X-Figma-Token": state['figma_access_token']}
            try:
                response = requests.get(url, headers=headers)
                response.raise_for_status()
                image_urls = response.json().get("images", {})
                logger.info(f"Fetched image URLs for {len(image_urls)} frames")
            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to fetch frame image URLs: {str(e)}")
        
        node_ids = list(state['image_nodes'].keys())
        if node_ids:
            url = f"https://api.figma.com/v1/images/{state['file_key']}?ids={','.join(node_ids)}&format=jpg"
            headers = {"X-Figma-Token": state['figma_access_token']}
            try:
                response = requests.get(url, headers=headers)
                response.raise_for_status()
                node_image_urls = response.json().get("images", {})
                logger.info(f"Fetched image URLs for {len(node_image_urls)} nodes")
            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to fetch node image URLs: {str(e)}")
        return {"image_urls": image_urls, "node_image_urls": node_image_urls}

    def _download_temp_image(self, url: str) -> str:
        temp_path = f"temp_{datetime.now().timestamp()}.jpg"
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content))
            max_size = (256, 256)
            img.thumbnail(max_size, Image.LANCZOS)
            img = img.convert("RGB")
            img.save(temp_path, "JPEG", quality=70, optimize=True)
            return temp_path
        except Exception as e:
            logger.error(f"Failed to process image: {str(e)}")
            raise

    def _encode_image(self, image_path: str) -> str:
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"Failed to encode image {image_path}: {str(e)}")
            return ""

    def _extract_metadata_details(self, frame_data: Dict) -> str:
        details = []
        details.append(f"Node Type: {frame_data.get('type', 'unknown')}")
        details.append(f"Node Name: {frame_data.get('name', 'unknown')}")
        if "absoluteBoundingBox" in frame_data:
            box = frame_data["absoluteBoundingBox"]
            x = float(box.get('x', 0)) if isinstance(box.get('x'), (int, float, str)) and str(box.get('x')).replace('.', '', 1).isdigit() else 0
            y = float(box.get('y', 0)) if isinstance(box.get('y'), (int, float, str)) and str(box.get('y')).replace('.', '', 1).isdigit() else 0
            width = float(box.get('width', 0)) if isinstance(box.get('width'), (int, float, str)) and str(box.get('width')).replace('.', '', 1).isdigit() else 0
            height = float(box.get('height', 0)) if isinstance(box.get('height'), (int, float, str)) and str(box.get('height')).replace('.', '', 1).isdigit() else 0
            details.append(f"Position: x={x}px, y={y}px")
            details.append(f"Size: width={width}px, height={height}px")
        if "style" in frame_data:
            style = frame_data["style"]
            details.append(f"Font: {style.get('fontFamily', 'unknown')} {style.get('fontSize', 0)}px")
            details.append(f"Font Weight: {style.get('fontWeight', 'normal')}")
            details.append(f"Line Height: {style.get('lineHeightPx', 'unknown')}px")
            details.append(f"Letter Spacing: {style.get('letterSpacing', 'unknown')}px")
        if "fills" in frame_data and frame_data["fills"]:
            for fill in frame_data["fills"]:
                if fill.get("type") == "SOLID":
                    color = fill.get("color", {})
                    hex_color = self._rgb_to_hex(
                        float(color.get('r', 0)) if isinstance(color.get('r'), (int, float, str)) and str(color.get('r')).replace('.', '', 1).isdigit() else 0,
                        float(color.get('g', 0)) if isinstance(color.get('g'), (int, float, str)) and str(color.get('g')).replace('.', '', 1).isdigit() else 0,
                        float(color.get('b', 0)) if isinstance(color.get('b'), (int, float, str)) and str(color.get('b')).replace('.', '', 1).isdigit() else 0
                    )
                    details.append(f"Fill Color: #{hex_color}")
                elif fill.get("type") == "IMAGE" and "imageRef" in fill:
                    details.append(f"Image Reference: {fill['imageRef']}")
        if "strokes" in frame_data and frame_data["strokes"]:
            stroke = frame_data["strokes"][0].get("color", {})
            hex_color = self._rgb_to_hex(
                float(stroke.get('r', 0)) if isinstance(stroke.get('r'), (int, float, str)) and str(stroke.get('r')).replace('.', '', 1).isdigit() else 0,
                float(stroke.get('g', 0)) if isinstance(stroke.get('g'), (int, float, str)) and str(stroke.get('g')).replace('.', '', 1).isdigit() else 0,
                float(stroke.get('b', 0)) if isinstance(stroke.get('b'), (int, float, str)) and str(stroke.get('b')).replace('.', '', 1).isdigit() else 0
            )
            details.append(f"Border Color: #{hex_color}")
            details.append(f"Border Width: {frame_data.get('strokeWeight', 0)}px")
        if "effects" in frame_data and frame_data["effects"]:
            for effect in frame_data["effects"]:
                if effect.get("type") == "DROP_SHADOW":
                    color = effect.get("color", {})
                    hex_color = self._rgb_to_hex(
                        float(color.get('r', 0)) if isinstance(color.get('r'), (int, float, str)) and str(color.get('r')).replace('.', '', 1).isdigit() else 0,
                        float(color.get('g', 0)) if isinstance(color.get('g'), (int, float, str)) and str(color.get('g')).replace('.', '', 1).isdigit() else 0,
                        float(color.get('b', 0)) if isinstance(color.get('b'), (int, float, str)) and str(color.get('b')).replace('.', '', 1).isdigit() else 0
                    )
                    details.append(f"Shadow: color=#{hex_color}, offsetX={effect.get('offset', {}).get('x', 0)}px, offsetY={effect.get('offset', {}).get('y', 0)}px, radius={effect.get('radius', 0)}px")
        if "zIndex" in frame_data:
            details.append(f"Z-Index: {frame_data.get('zIndex', 0)}")
        return "\n".join(details)

    def _rgb_to_hex(self, r: float, g: float, b: float) -> str:
        return f"{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"

    def _build_layout_tree(self, frame_data: Dict) -> str:
        def extract_essential_details(node: Dict, depth: int = 0) -> List[str]:
            details = []
            indent = "  " * depth
            node_type = node.get("type", "unknown")
            node_name = node.get("name", "unknown")
            if "absoluteBoundingBox" in node:
                box = node["absoluteBoundingBox"]
                x = float(box.get('x', 0)) if isinstance(box.get('x'), (int, float, str)) and str(box.get('x')).replace('.', '', 1).isdigit() else 0
                y = float(box.get('y', 0)) if isinstance(box.get('y'), (int, float, str)) and str(box.get('y')).replace('.', '', 1).isdigit() else 0
                width = float(box.get('width', 0)) if isinstance(box.get('width'), (int, float, str)) and str(box.get('width')).replace('.', '', 1).isdigit() else 0
                height = float(box.get('height', 0)) if isinstance(box.get('height'), (int, float, str)) and str(box.get('height')).replace('.', '', 1).isdigit() else 0
                details.append(f"{indent}- Node: {node_name} ({node_type})")
                details.append(f"{indent}  Position: x={x}px, y={y}px")
                details.append(f"{indent}  Size: width={width}px, height={height}px")
            if "style" in node:
                style = node["style"]
                details.append(f"{indent}  Font: {style.get('fontFamily', 'unknown')} {style.get('fontSize', 0)}px")
                details.append(f"{indent}  Font Weight: {style.get('fontWeight', 'normal')}")
                details.append(f"{indent}  Line Height: {style.get('lineHeightPx', 'unknown')}px")
                details.append(f"{indent}  Letter Spacing: {style.get('letterSpacing', 'unknown')}px")
            if "fills" in node and node["fills"]:
                for fill in node["fills"]:
                    if fill.get("type") == "SOLID":
                        color = fill.get("color", {})
                        hex_color = self._rgb_to_hex(
                            float(color.get('r', 0)) if isinstance(color.get('r'), (int, float, str)) and str(color.get('r')).replace('.', '', 1).isdigit() else 0,
                            float(color.get('g', 0)) if isinstance(color.get('g'), (int, float, str)) and str(color.get('g')).replace('.', '', 1).isdigit() else 0,
                            float(color.get('b', 0)) if isinstance(color.get('b'), (int, float, str)) and str(color.get('b')).replace('.', '', 1).isdigit() else 0
                        )
                        details.append(f"{indent}  Background Color: #{hex_color}")
                    elif fill.get("type") == "IMAGE" and "imageRef" in fill:
                        details.append(f"{indent}  Image Reference: {fill['imageRef']}")
            if "strokes" in node and node["strokes"]:
                stroke = node["strokes"][0].get("color", {})
                hex_color = self._rgb_to_hex(
                    float(stroke.get('r', 0)) if isinstance(stroke.get('r'), (int, float, str)) and str(stroke.get('r')).replace('.', '', 1).isdigit() else 0,
                    float(stroke.get('g', 0)) if isinstance(stroke.get('g'), (int, float, str)) and str(stroke.get('g')).replace('.', '', 1).isdigit() else 0,
                    float(stroke.get('b', 0)) if isinstance(stroke.get('b'), (int, float, str)) and str(stroke.get('b')).replace('.', '', 1).isdigit() else 0
                )
                details.append(f"{indent}  Border Color: #{hex_color}")
                details.append(f"{indent}  Border Width: {node.get('strokeWeight', 0)}px")
            if "effects" in node and node["effects"]:
                for effect in node["effects"]:
                    if effect.get("type") == "DROP_SHADOW":
                        color = effect.get("color", {})
                        hex_color = self._rgb_to_hex(
                            float(color.get('r', 0)) if isinstance(color.get('r'), (int, float, str)) and str(color.get('r')).replace('.', '', 1).isdigit() else 0,
                            float(color.get('g', 0)) if isinstance(color.get('g'), (int, float, str)) and str(color.get('g')).replace('.', '', 1).isdigit() else 0,
                            float(color.get('b', 0)) if isinstance(color.get('b'), (int, float, str)) and str(color.get('b')).replace('.', '', 1).isdigit() else 0
                        )
                        details.append(f"{indent}  Shadow: color=#{hex_color}, offsetX={effect.get('offset', {}).get('x', 0)}px, offsetY={effect.get('offset', {}).get('y', 0)}px, radius={effect.get('radius', 0)}px")
            if "children" in node:
                for child in node["children"]:
                    details.extend(extract_essential_details(child, depth + 1))
            return details
        return "\n".join(extract_essential_details(frame_data))

    def _process_frame(self, state: State) -> Dict:
        if state['current_frame_index'] >= len(state['figma_frames']):
            return {}
        frame = state['figma_frames'][state['current_frame_index']]
        frame_id = frame.get("id")
        screen_name = frame["name"].replace(" ", "")
        image_url = state['image_urls'].get(frame_id)
        frame_content = state['frame_content']
        if image_url:
            try:
                temp_image_path = self._download_temp_image(image_url)
                base64_image = self._encode_image(temp_image_path)
                layout_tree = self._build_layout_tree(frame)
                metadata_str = self._extract_metadata_details(frame)
                frame_images = {k: v for k, v in state['image_nodes'].items() if v.get("frame_id") == frame_id}
                image_metadata = "\n".join(
                    f"Image Node ID: {node_id}, Name: {data['name']}, URL: {state['node_image_urls'].get(node_id, 'Not available')}"
                    for node_id, data in frame_images.items()
                )
                metadata_str += f"\n\nFrame Images:\n{image_metadata}" if image_metadata else ""
                prompt = f'''
                Analyze this Figma mobile UI screen in 4 sections:
                1. Components
                    - List top-down.
                    - Type: Text, Image, Button, TextInput.
                    - For Images: include asset name from metadata.
                    - Mark interactive elements.
                    - Note nesting.
                2. Layout
                    - Show parent-child structure.
                    - Give x/y, width/height.
                    - Layout: Flex/Absolute.
                    - Alignment: row/column, spacing, margins/paddings.
                    - Describe relative positions.  
                3. Styles
                    For each:
                    - Colors (hex), font (family, size, weight, lineHeight)
                    - Borders, radius, shadow (values)
                    Alignment:
                    - textAlign, justifyContent, alignItems
                    Inputs: placeholder, text color, border
                    Images: resizeMode, aspectRatio, asset name
                4. Interactions
                    List tap/click actions.
                    Show navigation targets only if stated.
                    Note state changes (e.g., toggles, error).
                    Use:
                    Layout: {layout_tree}
                    Metadata: {metadata_str}
                    NOTE: the explanation should be clear and more informative and more shorter.
                '''
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        response = self.client.chat.completions.create(
                            model=self.model_vlm,
                            messages=[
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "text", "text": prompt},
                                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                                    ]
                                }
                            ],
                            temperature=0.6,
                            max_tokens=8192,
                            top_p=1,
                            stream=False
                        )
                        frame_content[screen_name] = {
                            "frame_id": frame_id,
                            "analysis": response.choices[0].message.content.strip(),
                            "metadata": metadata_str,
                            "image_url": image_url,
                            "node_images": {node_id: state['node_image_urls'].get(node_id) for node_id in frame_images}
                        }
                        os.remove(temp_image_path)
                        logger.info(f"Processed frame: {screen_name}")
                        break    
                    except Exception as e:
                        if "rate limit" in str(e).lower() and attempt < max_retries - 1:
                            wait_time = 2 ** attempt
                            logger.warning(f"Rate limit hit for {screen_name}. Retrying after {wait_time}s...")
                            time.sleep(wait_time)
                        else:
                            logger.error(f"Error processing frame {screen_name}: {str(e)}")
                            frame_content[screen_name] = {
                                "frame_id": frame_id,
                                "analysis": "",
                                "metadata": metadata_str,
                                "image_url": image_url,
                                "node_images": {node_id: state['node_image_urls'].get(node_id) for node_id in frame_images},
                                "error": str(e)
                            }
                            os.remove(temp_image_path)
                            break
            except Exception as e:
                logger.error(f"Failed to process frame {screen_name}: {str(e)}")
                frame_content[screen_name] = {
                    "frame_id": frame_id,
                    "analysis": "",
                    "metadata": self._extract_metadata_details(frame),
                    "image_url": image_url,
                    "node_images": {node_id: state['node_image_urls'].get(node_id) for node_id in frame_images},
                    "error": str(e)
                }
        else:
            logger.warning(f"No image URL for frame {screen_name}")
            frame_content[screen_name] = {
                "frame_id": frame_id,
                "analysis": "",
                "metadata": self._extract_metadata_details(frame),
                "image_url": "",
                "node_images": {node_id: state['node_image_urls'].get(node_id) for node_id in frame_images},
                "error": "No image URL available"
            }
        return {"frame_content": frame_content, "current_frame_index": state['current_frame_index'] + 1}

    def _check_more_frames(self, state: State) -> str:
        return "next_frame" if state['current_frame_index'] < len(state['figma_frames']) else "done"

    def _verify_content(self, state: State) -> Dict:
        frame_content = state['frame_content']
        for screen_name, content in frame_content.items():
            if not content.get("analysis"):
                logger.warning(f"Verification failed for {screen_name}: No analysis content")
                try:
                    frame = next(f for f in state['figma_frames'] if f["name"].replace(" ", "") == screen_name)
                    image_url = state['image_urls'].get(frame["id"])
                    if image_url:
                        temp_image_path = self._download_temp_image(image_url)
                        base64_image = self._encode_image(temp_image_path)
                        layout_tree = self._build_layout_tree(frame)
                        metadata_str = content["metadata"]
                        prompt = f'''
                        Previous analysis failed. Retry analyzing the Figma frame for {screen_name}.
                        Provide analysis in the same 4 sections as before:

                        Section 1: Component Identification
                        Section 2: Layout Structure
                        Section 3: Style Information
                        Section 4: Interactions (if any)

                        Essential Layout and Style Details:
                        {layout_tree}

                        Frame Metadata:
                        {metadata_str}
                        '''
                        max_retries = 3
                        for attempt in range(max_retries):
                            try:
                                response = self.client.chat.completions.create(
                                    model=self.model_vlm,
                                    messages=[
                                        {
                                            "role": "user",
                                            "content": [
                                                {"type": "text", "text": prompt},
                                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                                            ]
                                        }
                                    ],
                                    temperature=0.7,
                                    max_tokens=8192,
                                    top_p=1,
                                    stream=False
                                )
                                frame_content[screen_name] = {
                                    **content,
                                    "analysis": response.choices[0].message.content.strip(),
                                    "error": ""
                                }
                                os.remove(temp_image_path)
                                logger.info(f"Successfully reprocessed frame: {screen_name}")
                                break
                            except Exception as e:
                                if "rate limit" in str(e).lower() and attempt < max_retries - 1:
                                    wait_time = 2 ** attempt
                                    logger.warning(f"Rate limit hit for {screen_name} during verification. Retrying after {wait_time}s...")
                                    time.sleep(wait_time)
                                else:
                                    logger.error(f"Failed to reprocess frame {screen_name}: {str(e)}")
                                    frame_content[screen_name] = {
                                        **content,
                                        "error": str(e)
                                    }
                                    os.remove(temp_image_path)
                                    break
                    else:
                        logger.error(f"Cannot reprocess {screen_name}: No image URL")
                except Exception as e:
                    logger.error(f"Failed to reprocess frame {screen_name}: {str(e)}")
                    frame_content[screen_name] = {
                        **content,
                        "error": str(e)
                    }
        return {"frame_content": frame_content}

    def _compile_results(self, state: State) -> Dict:
        output = {
            "project_name": "figma_analysis",
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "frames": state['frame_content']
        }
        with open("figma_frame_analysis.json", "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        logger.info("Compiled results saved to figma_frame_analysis.json")
        return {}

    def analyze_frames(self, file_key: str) -> Dict:
        state = {
            "figma_frames": [],
            "image_nodes": {},
            "image_urls": {},
            "node_image_urls": {},
            "frame_content": {},
            "current_frame_index": 0,
            "file_key": file_key,
            "figma_access_token": self.figma_access_token
        }
        result = self.graph.invoke(state)
        return {
            "success": bool(result['frame_content']),
            "frames": result['frame_content']
        }
    
if __name__ == "__main__":
    figma_access_token = "figd_HpiGFCGKNjKxpfMUvmjc8OruEpHVjcAcQ5q2tTRA"
    file_key = "21nQyQEPLLaQ8blSU0WxQm"

    analyzer = FigmaFrameAnalyzer(figma_access_token=figma_access_token)
    result = analyzer.analyze_frames(file_key=file_key)

    print(result)