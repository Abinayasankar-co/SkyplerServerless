import os
import base64
import json
import time
import shutil
import requests
import re
from groq import Groq
import streamlit as st
from typing import Dict, List, Optional
from dotenv import load_dotenv
from datetime import datetime
from PIL import Image  
from io import BytesIO
from logger import logging

load_dotenv()


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

class CodeGenerator:
    def __init__(self, figma_access_token: str, 
                 model_vlm: str = "meta-llama/llama-4-scout-17b-16e-instruct", #"meta-llama/llama-4-maverick-17b-128e-instruct",#"meta-llama/llama-4-scout-17b-16e-instruct", 
                 model_llm: str = "deepseek-r1-distill-llama-70b"):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.figma_access_token = figma_access_token
        self.model_vlm = model_vlm
        self.model_llm = model_llm
        self.project_base_dir = "generated_projects"
        self.project_name: Optional[str] = None
        self.asset_map: Dict[str, str] = {}  # Maps node ID or name to asset path
        self.figma_frames: List[Dict] = []
        self.image_nodes: Dict[str, Dict] = {}  # Maps node ID to image node data
        self._image_counter = 0  # Track generic image names for frames

    def validate_figma_access(self, file_key: str) -> bool:
        """Validate Figma access token and file key."""
        url = f"https://api.figma.com/v1/files/{file_key}"
        headers = {"X-Figma-Token": self.figma_access_token}
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            return True
        except requests.exceptions.HTTPError as e:
            logger.error(f"Figma API error: {str(e)}")
            if response.status_code == 403:
                logger.error("Invalid or unauthorized Figma access token.")
            elif response.status_code == 404:
                logger.error("Figma file key not found. Please check the file key.")
            return False
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to validate Figma access: {str(e)}")
            return False
    
    def find_frames_recursive(self, nodes: List[Dict], depth: int = 0, max_depth: int = 1) -> List[Dict]:
        """Find only top-level frames in the node tree (directly under CANVAS)."""
        frames = []
        for node in nodes:
            if node.get("type") == "FRAME" and depth == 0:
                frames.append(node)
            if depth < max_depth and "children" in node:
                frames.extend(self.find_frames_recursive(node["children"], depth + 1, max_depth))
        return frames
    
    def find_image_nodes(self, nodes: List[Dict], frame_id: str = None) -> Dict[str, Dict]:
        """Recursively find image nodes and their URLs in the Figma JSON."""
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
                image_nodes.update(self.find_image_nodes(node["children"], frame_id))
        return image_nodes

    def fetch_figma_frames(self, file_key: str) -> List[Dict]:
        """Fetch Figma frames and extract image nodes from the JSON."""
        if not self.validate_figma_access(file_key):
            return []
        
        url = f"https://api.figma.com/v1/files/{file_key}"
        headers = {"X-Figma-Token": self.figma_access_token}
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()
            logger.info("Figma API response received.")
            
            document = data.get("document", {})
            children = document.get("children", [])
            logger.info(f"Found {len(children)} top-level nodes in Figma file.")
            
            self.figma_frames = self.find_frames_recursive(children)
            logger.info("Detected frames: " + ", ".join([frame["name"] for frame in self.figma_frames if "name" in frame]))
            
            # Extract image nodes from each frame
            for frame in self.figma_frames:
                frame_id = frame.get("id")
                if "children" in frame:
                    self.image_nodes.update(self.find_image_nodes(frame["children"], frame_id))
            
            if not self.figma_frames:
                logger.warning("No FRAME nodes found. Checking for CANVAS nodes...")
                for node in children:
                    if node.get("type") == "CANVAS" and "children" in node:
                        self.figma_frames.extend(self.find_frames_recursive(node["children"]))
                        for frame in self.figma_frames:
                            frame_id = frame.get("id")
                            if "children" in frame:
                                self.image_nodes.update(self.find_image_nodes(frame["children"], frame_id))
                        logger.info("Detected frames after CANVAS: " + ", ".join([frame["name"] for frame in self.figma_frames if "name" in frame]))
            
            logger.info(f"Total frames found: {len(self.figma_frames)}")
            logger.info(f"Total image nodes found: {len(self.image_nodes)}")
            if not self.figma_frames:
                logger.error("No frames found in the Figma file. Ensure the file contains frames or canvases with frames.")
            
            return self.figma_frames
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch Figma frames: {str(e)}")
            return []

    def fetch_figma_image_urls(self, file_key: str, node_ids: List[str]) -> Dict[str, str]:
        """Fetch image URLs for given node IDs."""
        if not node_ids:
            logger.warning("No node IDs provided for image URL fetching.")
            return {}
        url = f"https://api.figma.com/v1/images/{file_key}?ids={','.join(node_ids)}&format=jpg"
        headers = {"X-Figma-Token": self.figma_access_token}
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            image_urls = response.json().get("images", {})
            logger.info(f"Fetched image URLs for {len(image_urls)} nodes.")
            return image_urls
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch image URLs: {str(e)}")
            return {}
    
    def download_image(self, url: str, asset_name: str, subfolder: str = "images", counter: int = 0, is_frame: bool = False) -> str:
        """Download an image and save it to the project's assets subfolder with unique name."""
        project_folder = os.path.join(self.project_base_dir, self.project_name)
        
        if is_frame and subfolder == "images":
            # For frame images in images folder, use generic name like image1.jpg
            base_asset_name = f"image{self._image_counter + 1}"
            ext = ".jpg"
            self._image_counter += 1
        else:
            # For frames folder or non-frame images, use provided asset_name
            base_asset_name, ext = os.path.splitext(asset_name)
        
        unique_asset_name = f"{base_asset_name}{f'_{counter}' if counter > 0 else ''}{ext}"
        asset_path = os.path.join(project_folder, "src/assets", subfolder, unique_asset_name)
        
        # Check if file already exists
        if os.path.exists(asset_path):
            return self.download_image(url, asset_name, subfolder, counter + 1, is_frame)
        
        os.makedirs(os.path.dirname(asset_path), exist_ok=True)
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(asset_path, "wb") as f:
                shutil.copyfileobj(response.raw, f)
            return f"../assets/{subfolder}/{unique_asset_name}"
        except requests.exceptions.RequestException as e:
            logger.warning(f"Failed to save asset {unique_asset_name}: {str(e)}")
            return ""

    def encode_image(self, image_path: str) -> str:
        """Encode an image file to base64."""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"Failed to encode image {image_path}: {str(e)}")
            return ""

    def download_temp_image(self, url: str) -> str:
        """Download an image to a temporary file, resize, and compress it."""
        temp_path = f"temp_{time.time()}.jpg"
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # Load image into Pillow
            img = Image.open(BytesIO(response.content))
            
            # Resize image to a maximum of 512x512 while preserving aspect ratio
            max_size = (512, 512)
            img.thumbnail(max_size, Image.LANCZOS)  # LANCZOS for high-quality resizing
            
            # Save compressed image to temporary file
            img.save(temp_path, "JPEG", quality=85, optimize=True)  # Quality=85 for good balance
            return temp_path
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download temporary image: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Failed to process image: {str(e)}")
            raise
    
    def extract_metadata_details(self, frame_data: Dict) -> str:
        """Extract detailed styling and positioning from Figma metadata."""
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
                    hex_color = self.rgb_to_hex(
                        float(color.get('r', 0)) if isinstance(color.get('r'), (int, float, str)) and str(color.get('r')).replace('.', '', 1).isdigit() else 0,
                        float(color.get('g', 0)) if isinstance(color.get('g'), (int, float, str)) and str(color.get('g')).replace('.', '', 1).isdigit() else 0,
                        float(color.get('b', 0)) if isinstance(color.get('b'), (int, float, str)) and str(color.get('b')).replace('.', '', 1).isdigit() else 0
                    )
                    details.append(f"Fill Color: #{hex_color}")
                elif fill.get("type") == "IMAGE" and "imageRef" in fill:
                    details.append(f"Image Reference: {fill['imageRef']}")
        if "strokes" in frame_data and frame_data["strokes"]:
            stroke = frame_data["strokes"][0].get("color", {})
            hex_color = self.rgb_to_hex(
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
                    hex_color = self.rgb_to_hex(
                        float(color.get('r', 0)) if isinstance(color.get('r'), (int, float, str)) and str(color.get('r')).replace('.', '', 1).isdigit() else 0,
                        float(color.get('g', 0)) if isinstance(color.get('g'), (int, float, str)) and str(color.get('g')).replace('.', '', 1).isdigit() else 0,
                        float(color.get('b', 0)) if isinstance(color.get('b'), (int, float, str)) and str(color.get('b')).replace('.', '', 1).isdigit() else 0
                    )
                    details.append(f"Shadow: color=#{hex_color}, offsetX={effect.get('offset', {}).get('x', 0)}px, offsetY={effect.get('offset', {}).get('y', 0)}px, radius={effect.get('radius', 0)}px")
        if "zIndex" in frame_data:
            details.append(f"Z-Index: {frame_data.get('zIndex', 0)}")
        return "\n".join(details)

    def rgb_to_hex(self, r: float, g: float, b: float) -> str:
        """Convert RGB (0-1) to hex color."""
        return f"{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"

    def build_layout_tree(self, frame_data: Dict) -> str:
        """Extract essential positioning and CSS details from frame data."""
        def extract_essential_details(node: Dict, depth: int = 0) -> List[str]:
            details = []
            indent = "  " * depth
            node_type = node.get("type", "unknown")
            node_name = node.get("name", "unknown")
            
            # Positioning
            if "absoluteBoundingBox" in node:
                box = node["absoluteBoundingBox"]
                x = float(box.get('x', 0)) if isinstance(box.get('x'), (int, float, str)) and str(box.get('x')).replace('.', '', 1).isdigit() else 0
                y = float(box.get('y', 0)) if isinstance(box.get('y'), (int, float, str)) and str(box.get('y')).replace('.', '', 1).isdigit() else 0
                width = float(box.get('width', 0)) if isinstance(box.get('width'), (int, float, str)) and str(box.get('width')).replace('.', '', 1).isdigit() else 0
                height = float(box.get('height', 0)) if isinstance(box.get('height'), (int, float, str)) and str(box.get('height')).replace('.', '', 1).isdigit() else 0
                details.append(f"{indent}- Node: {node_name} ({node_type})")
                details.append(f"{indent}  Position: x={x}px, y={y}px")
                details.append(f"{indent}  Size: width={width}px, height={height}px")
            
            # CSS Details
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
                        hex_color = self.rgb_to_hex(
                            float(color.get('r', 0)) if isinstance(color.get('r'), (int, float, str)) and str(color.get('r')).replace('.', '', 1).isdigit() else 0,
                            float(color.get('g', 0)) if isinstance(color.get('g'), (int, float, str)) and str(color.get('g')).replace('.', '', 1).isdigit() else 0,
                            float(color.get('b', 0)) if isinstance(color.get('b'), (int, float, str)) and str(color.get('b')).replace('.', '', 1).isdigit() else 0
                        )
                        details.append(f"{indent}  Background Color: #{hex_color}")
                    elif fill.get("type") == "IMAGE" and "imageRef" in fill:
                        details.append(f"{indent}  Image Reference: {fill['imageRef']}")
            
            if "strokes" in node and node["strokes"]:
                stroke = node["strokes"][0].get("color", {})
                hex_color = self.rgb_to_hex(
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
                        hex_color = self.rgb_to_hex(
                            float(color.get('r', 0)) if isinstance(color.get('r'), (int, float, str)) and str(color.get('r')).replace('.', '', 1).isdigit() else 0,
                            float(color.get('g', 0)) if isinstance(color.get('g'), (int, float, str)) and str(color.get('g')).replace('.', '', 1).isdigit() else 0,
                            float(color.get('b', 0)) if isinstance(color.get('b'), (int, float, str)) and str(color.get('b')).replace('.', '', 1).isdigit() else 0
                        )
                        details.append(f"{indent}  Shadow: color=#{hex_color}, offsetX={effect.get('offset', {}).get('x', 0)}px, offsetY={effect.get('offset', {}).get('y', 0)}px, radius={effect.get('radius', 0)}px")
            
            # Recursively process children
            if "children" in node:
                for child in node["children"]:
                    details.extend(extract_essential_details(child, depth + 1))
            
            return details
        
        return "\n".join(extract_essential_details(frame_data))

    def analyze_figma_frame(self, image_url: str, frame_data: Optional[Dict] = None) -> str:
        """Analyze a Figma frame, including image metadata."""
        temp_image_path = self.download_temp_image(image_url)
        base64_image = self.encode_image(temp_image_path)

        layout_tree = self.build_layout_tree(frame_data) if frame_data else ""
        metadata_str = self.extract_metadata_details(frame_data) if frame_data else ""
        frame_id = frame_data.get("id") if frame_data else None
        
        # Include image nodes for this frame
        frame_images = {k: v for k, v in self.image_nodes.items() if v.get("frame_id") == frame_id}
        image_metadata = "\n".join(
            f"Image Node ID: {node_id}, Name: {data['name']}, Asset Path: {self.asset_map.get(node_id, 'Not downloaded')}"
            for node_id, data in frame_images.items()
        )
        # Add frame image metadata
        frame_image_path = self.asset_map.get(frame_id, 'Not downloaded')
        image_metadata = f"Frame Image: {frame_data.get('name', 'unknown')}, Asset Path: {frame_image_path}\n{image_metadata}" if frame_image_path else image_metadata
        metadata_str += f"\n\nFrame Images:\n{image_metadata}" if image_metadata else ""

        with open(os.path.join(self.project_base_dir, self.project_name, f"debug_metadata_{frame_data.get('name', 'unknown').replace(' ', '')}.log"), "w") as f:
            f.write(metadata_str)

        prompt = f'''
        You are analyzing an app screen designed in Figma. The image represents a mobile screen UI.
        Provide your analysis in the following 4 sections:

        Section 1: Component Identification
        - List all components in top-down order.
        - Identify type (e.g., Text, Image, Button, TextInput).
        - For Image components, specify the corresponding asset name from the Frame Images metadata (e.g., image1.jpg for the frame image, settings_icon.jpg for icons).
        - Indicate if components are nested within other components.
        - Clearly mark interactive components (buttons, inputs, links).

        Section 2: Layout Structure
        - Describe parent-child hierarchy of components.
        - Include absolute position (x/y) and dimensions (width/height).
        - Mention layout technique used (e.g., flexbox with column layout).
        - Describe alignment: Are components in row/column? Are they evenly spaced?
        - If spacing is consistent, state padding/margin values.
        - Note how components are positioned relative to each other (not just absolutely).

        Section 3: Style Information
        - For each component, specify:
        - Colors (background/text) with exact hex values when possible
        - Font details (fontFamily, fontSize, lineHeight)
        - Border/shadow properties with specific values
        - Alignments (textAlign, alignItems, justifyContent)
        - resizeMode and aspectRatio for images
        - Padding and margin values
        - For input fields, specify placeholder text, text color, and border styles
        - For Image components, ensure the source matches the asset name provided in Frame Images metadata (e.g., image1.jpg for the frame)

        Section 4: Interactions (if any)
        - Mention if a component triggers navigation or action (e.g., buttons).
        - Specify target screen ONLY if the text explicitly indicates navigation (e.g., "Login" or "Next" with a clear target screen name like "Home" or "Dashboard").
        - Do NOT assume navigation for buttons with generic actions (e.g., "Submit", "STATIC", "DDNS") unless a specific target screen is mentioned.
        - Note any state changes (like toggle visibility for password fields).

        ---

        Essential Layout and Style Details:
        {layout_tree}

        Frame Metadata:
        {metadata_str}
        '''
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
                analysis = response.choices[0].message.content.strip()
                os.remove(temp_image_path)
                return analysis
        except Exception as e:
                logger.error(f"VLM analysis failed: {str(e)}")
                if os.path.exists(temp_image_path):
                    os.remove(temp_image_path)
                return ""

    def compare_images(self, figma_image_path: str, user_image_path: str) -> str:
        """Compare Figma frame image with user-uploaded output image."""
        figma_base64 = self.encode_image(figma_image_path)
        user_base64 = self.encode_image(user_image_path)

        prompt = '''
        You are an expert UI analyst comparing two images: a Figma design (reference) and a generated app output (user-uploaded). 
        Analyze both images and provide a detailed comparison in the following sections:

        Section 1: Component Differences
        - List components present in the Figma design but missing or incorrect in the user output.
        - Note any additional components in the user output not in the Figma design.

        Section 2: Layout Discrepancies
        - Compare the parent-child hierarchy and positioning (x/y, width/height).
        - Identify misaligned components or incorrect spacing (padding/margin).
        - Note differences in flexbox layout (row/column, alignment).

        Section 3: Styling Issues
        - Compare colors (background/text) and note any mismatches (provide hex values).
        - Check font details (fontFamily, fontSize, lineHeight) for discrepancies.
        - Identify differences in border/shadow properties.
        - Note incorrect alignments (textAlign, alignItems, justifyContent).
        - Check image properties (resizeMode, aspectRatio).
        - For input fields, compare placeholder text, text color, and border styles.

        Section 4: Interaction Issues
        - Compare interactive components (buttons, inputs) and their actions.
        - Note missing or incorrect navigation targets or state changes (e.g., password visibility toggle).

        Provide specific instructions for changes needed to make the user output match the Figma design exactly.
        '''
        try:
            response = self.client.chat.completions.create(
                model=self.model_vlm,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{figma_base64}"}},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{user_base64}"}}
                        ]
                    }
                ],
                temperature=0.6,
                max_tokens=8192,
                top_p=1,
                stream=False
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            st.error(f"Image comparison failed: {str(e)}")
            return ""

    def rework_interface(self, project_name: str, language_type: str):
        """Display rework interface with Figma frames and upload option."""
        self.project_name = project_name
        project_folder = os.path.join(self.project_base_dir, project_name)
        frames_folder = os.path.join(project_folder, "src/assets/frames")
        
        if not os.path.exists(frames_folder):
            st.error("Frames folder not found. Please generate the project first.")
            return

        # Get list of Figma frame images
        frame_images = [f for f in os.listdir(frames_folder) if f.endswith('.jpg')]
        if not frame_images:
            st.warning("No Figma frame images found in frames folder.")
            return

        st.subheader("Compare and Refine Generated Output")
        for frame_image in frame_images:
            screen_name = frame_image.replace('.jpg', '')
            st.markdown(f"### Screen: {screen_name}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Figma Design**")
                image_path = os.path.join(frames_folder, frame_image)
                st.image(image_path, use_container_width=True)

            with col2:
                st.markdown("**Current Output**")
                uploaded_file = st.file_uploader(f"Upload output for {screen_name}", type=['jpg', 'png'], key=f"upload_{screen_name}")
                
                if uploaded_file:
                    # Save uploaded image temporarily
                    temp_dir = os.path.join(project_folder, "temp")
                    os.makedirs(temp_dir, exist_ok=True)
                    temp_path = os.path.join(temp_dir, f"{screen_name}_output.jpg")
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    st.image(temp_path, use_container_width=True)
                    
                    if st.button("Analyze and Fix", key=f"analyze_{screen_name}"):
                        st.info(f"Analyzing differences for {screen_name}...")
                        comparison_result = self.compare_images(image_path, temp_path)
                        
                        if comparison_result:
                            st.text_area("Comparison Result", comparison_result, height=300)
                            
                            # Get existing code
                            ext = "tsx" if language_type == "typescript" else "jsx"
                            code_path = os.path.join(project_folder, "src/screens", f"{screen_name}.{ext}")
                            if os.path.exists(code_path):
                                with open(code_path, "r", encoding="utf-8") as f:
                                    existing_code = f.read()
                                
                                # Generate updated code
                                updated_code = self.generate_updated_code(existing_code, comparison_result, screen_name, language_type)
                                if updated_code:
                                    self.save_generated_code(updated_code, screen_name, language_type)
                                    st.success(f"Updated code for {screen_name} saved successfully!")
                                    # Update session state
                                    st.session_state.generated_code = st.session_state.generated_code.replace(existing_code, updated_code)
                                else:
                                    st.error(f"Failed to generate updated code for {screen_name}")
                        else:
                            st.error(f"Failed to compare images for {screen_name}")

    def generate_updated_code(self, existing_code: str, comparison_result: str, screen_name: str, language_type: str) -> str:
        """Generate updated React Native code based on comparison result."""
        # Create a list of available assets
        available_assets = list(self.asset_map.values())
        assets_str = "\n".join([f"- {asset}" for asset in available_assets])

        prompt = f'''
        You are an expert React Native developer tasked with updating an existing component to match a Figma design exactly.
        Below is the existing code and the comparison result detailing differences between the Figma design and the current output.

        Existing Code:
        ```{language_type}
        {existing_code}
        ```

        Comparison Result:
        {comparison_result}

        Available Assets:
        {assets_str}

        Requirements:
        - Update the existing code to address all issues listed in the comparison result.
        - Maintain the same structure (functional component, StyleSheet, SafeAreaView).
        - Use React Native built-in components (View, Text, Image, TouchableOpacity, TextInput).
        - Do NOT use inline styles.
        - Ensure TextInput components have no child components.
        - For password inputs with visibility toggle, use a container View with absolute positioning for the toggle button.
        - Set image size using aspectRatio and resizeMode.
        - Add accessibilityLabel for interactive elements.
        - Use useNavigation only if navigation is required.
        - Add onPress handlers for all TouchableOpacity components.
        - Ensure proper state management with useState for interactive elements.
        - For Image components, use the exact asset paths from the Available Assets list (e.g., require('../assets/images/image1.jpg')).
        - Do NOT use placeholder images unless explicitly mentioned in the comparison result.
        - Return the complete updated code in the following format:

        ```{language_type}
        {{updated_code}}
        ```

        Constraints:
        - Only make changes necessary to fix the discrepancies listed in the comparison result.
        - Preserve any correct parts of the existing code.
        - Ensure no overlapping elements by using proper margins, padding, and flexbox layouts.
        - Use pixel-to-dp conversion (import Dimensions and multiply pixel values by (Dimensions.get('window').width / 375)) for responsive scaling.
        '''
        try:
            response = self.client.chat.completions.create(
                model=self.model_llm,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=8192,
                top_p=1,
                stream=False
            )
            content = response.choices[0].message.content
            start = content.find(f"{language_type}") + len(f"{language_type}") + 1
            end = content.rfind("```")
            code = content[start:end].strip()
            return self.validate_and_fix_code(code, language_type)
        except Exception as e:
            st.error(f"Failed to generate updated code for {screen_name}: {str(e)}")
            return ""

    def validate_and_fix_code(self, code: str, language_type: str) -> str:
        """Validate and fix generated code, focusing on imports and syntax."""
        project_folder = os.path.join(self.project_base_dir, self.project_name)
        os.makedirs(project_folder, exist_ok=True)
        with open(os.path.join(project_folder, "debug_validate_input.log"), "a", encoding="utf-8") as f:
            f.write(f"Input Code:\n{code}\n{'-'*50}\n")
        
        # Parse existing imports
        existing_imports = {}
        import_pattern = r'import\s+{([^}]*)}\s+from\s+[\'"]([^\'"]+)[\'"]\s*;'
        for match in re.finditer(import_pattern, code):
            modules = [m.strip() for m in match.group(1).split(',')]
            package = match.group(2)
            if package not in existing_imports:
                existing_imports[package] = set()
            existing_imports[package].update(modules)
        
        # Define required imports based on code content
        required_modules = {'react-native': {'StyleSheet', 'View', 'Text'}}
        uses_navigation = 'navigation.' in code or 'useNavigation' in code
        uses_dimensions = 'Dimensions.get' in code
        typescript = language_type == 'typescript'
        
        # Add modules based on code content
        if 'TouchableOpacity' in code:
            required_modules['react-native'].add('TouchableOpacity')
        if 'Image' in code:
            required_modules['react-native'].add('Image')
        if 'TextInput' in code:
            required_modules['react-native'].add('TextInput')
        if uses_dimensions:
            required_modules['react-native'].add('Dimensions')
        if typescript:
            required_modules['react-native'].update({'ViewStyle', 'TextStyle', 'ImageStyle'})
        if uses_navigation:
            required_modules['@react-navigation/native'] = {'useNavigation'}
            if typescript:
                required_modules['@react-navigation/core'] = {'NavigationProp'}
        
        # Check if navigation is unused
        if uses_navigation:
            nav_declaration = 'const navigation = useNavigation'
            nav_references = len(re.findall(r'\bnavigation\.', code))
            if nav_declaration in code and nav_references == 0:
                code = re.sub(r'import\s*{{\s*useNavigation\s*}}\s*from\s*[\'"]@react-navigation/native[\'"]\s*;\n?', '', code)
                if typescript:
                    code = re.sub(r'import\s*{{\s*NavigationProp\s*}}\s*from\s*[\'"]@react-navigation/core[\'"]\s*;\n?', '', code)
                code = re.sub(r'const navigation = useNavigation[^;]*;\n?', '', code)
                uses_navigation = False
        
        # Merge existing and required imports
        final_imports = {}
        for package, modules in required_modules.items():
            final_imports[package] = final_imports.get(package, set()).union(modules)
        for package, modules in existing_imports.items():
            final_imports[package] = final_imports.get(package, set()).union(modules)
        
        # Generate import statements
        import_statements = []
        for package, modules in final_imports.items():
            if package == '@react-navigation/native' and not uses_navigation:
                continue
            if package == '@react-navigation/core' and not uses_navigation:
                continue
            if modules:
                sorted_modules = sorted(modules)
                import_statements.append(f"import {{ {', '.join(sorted_modules)} }} from '{package}';")
        import_section = '\n'.join(import_statements)
        
        # Remove existing imports from code
        code = re.sub(import_pattern, '', code)
        code = re.sub(r'import\s+React\s+from\s+[\'"]react[\'"]\s*;', '', code)
        
        # Fix syntax errors
        code = re.sub(r'\bport\b', 'import', code)
        code = re.sub(r'rgb\((\d+),\s*(\d+),\s*(\d+)\)', lambda m: f'#{int(m.group(1)):02x}{int(m.group(2)):02x}{int(m.group(3)):02x}', code)
        
        # Ensure React import
        if not code.strip().startswith('import React'):
            import_section = f"import React from 'react';\n{import_section}"
        
        # Reconstruct code
        code_lines = code.split('\n')
        code_lines = [line for line in code_lines if line.strip()]
        code_body = '\n'.join(code_lines)
        code = f"{import_section}\n\n{code_body}".strip()
        
        # Remove duplicate useNavigation declarations
        if uses_navigation and typescript:
            code = re.sub(r'const navigation = useNavigation\(\);', 
                         r'const navigation = useNavigation<NavigationProp<any>>();', code)
            nav_count = len(re.findall(r'const navigation = useNavigation', code))
            if nav_count > 1:
                lines = code.split('\n')
                nav_lines = [i for i, line in enumerate(lines) if 'const navigation = useNavigation' in line]
                for i in nav_lines[:-1]:
                    lines[i] = ''
                code = '\n'.join(line for line in lines if line.strip())
        
        with open(os.path.join(project_folder, "debug_validate_output.log"), "a", encoding="utf-8") as f:
            f.write(f"Fixed Code:\n{code}\n{'-'*50}\n")
        
        return code

    def generate_react_native_code(self, analysis: str, screen_name: str, language_type: str) -> str:
        """Generate React Native code with actual image paths."""
        extension = "tsx" if language_type == "typescript" else "jsx"
        type_annotation = ": React.FC" if language_type == "typescript" else ""
        typescript_types = ", ViewStyle, TextStyle, ImageStyle" if language_type == "typescript" else ""

        # Create a list of available assets for this screen
        available_assets = list(self.asset_map.values())
        assets_str = "\n".join([f"- {asset}" for asset in available_assets])

        prompt = f"""
        You are an expert React Native developer.
        Generate a clean, highly accurate component in {language_type} based on the analysis below.

        Available Assets:
        {assets_str}

        Constraints:
        - Use only functional components with useState hook where needed for state management.
        - Use React Native built-in components: View, Text, Image, TouchableOpacity, TextInput, SafeAreaView.
        - Use Flexbox and StyleSheet for layout and styling.
        - Do NOT use inline styles.
        - TextInput components must NOT have child components - they cannot contain other elements.
        - For password inputs with toggle visibility, create a container View with the TextInput and a TouchableOpacity positioned absolutely.
        - Set up proper state management for interactive elements (e.g., useState for password visibility).
        - Match positioning, padding, fonts, and colors precisely.
        - Set image size using aspectRatio and resizeMode where applicable.
        - Name the component "{screen_name}" and export it as default.
        - Add proper import statements - include useNavigation from '@react-navigation/native' ONLY if the analysis explicitly mentions a navigation interaction (e.g., a button with text like "Login" or "Next" that triggers navigation to another screen, or a specific navigation action like navigation.navigate('ScreenName')). If no navigation interactions are mentioned, do NOT import or use useNavigation. For example, buttons with actions like console.log or state changes (e.g., toggle visibility) do NOT require useNavigation.
        - Add accessibilityLabel for all interactive elements.
        - Ensure all Image components use valid source properties from the Available Assets list (e.g., require('../assets/images/image1.jpg')).
        - Do NOT use placeholder images unless explicitly mentioned in the analysis.
        - Add placeholderTextColor for TextInput components.
        - Wrap main view in a SafeAreaView.
        - Add onPress handlers for all TouchableOpacity components (with console.log at minimum).
        - Use pixel-to-dp conversion (import Dimensions and multiply pixel values by (Dimensions.get('window').width / 375)) for responsive scaling.

        If components are in a vertical column, use flexDirection: 'column'. If they are side-by-side, use flexDirection: 'row'.

        Check for these common errors:
        1. TextInput components must NOT have child components.
        2. Every Image component must have a valid source from the Available Assets list.
        3. Components with state need to import and use useState.
        4. Password inputs with visibility toggle must use a container View with absolutely positioned button.
        5. All TouchableOpacity components should have onPress handlers.

        Example file structure:
        ```{language_type}
        import React, {{ useState }} from 'react';
        import {{ SafeAreaView, View, Text, StyleSheet, Image, TouchableOpacity, TextInput, Dimensions{typescript_types} }} from 'react-native';
        // Only include the following if navigation is explicitly required
        // import {{ useNavigation }} from '@react-navigation/native';
        // import {{ NavigationProp }} from '@react-navigation/core';

        const {screen_name}{type_annotation} = () => {{
            // Add state management where needed
            // const [password, setPassword] = useState('');
            // const [passwordVisible, setPasswordVisible] = useState(false);

            // Only include navigation if explicitly required
            // const navigation = useNavigation<NavigationProp<any>>();
            // Example navigation usage:
            // <TouchableOpacity 
            //     onPress={{() => navigation.navigate('NextScreen')}}
            //     accessibilityLabel="Navigate to NextScreen">
            //     <Text>Go to Next Screen</Text>
            // </TouchableOpacity>

            return (
                <SafeAreaView style={{styles.container}}>
                    {{/* Components go here */}}
                    <Image
                        source={{require('../assets/images/image1.jpg')}} // Use asset from Available Assets list
                        style={{styles.image}}
                        resizeMode="contain"
                        accessibilityLabel="{screen_name} image"
                    />
                </SafeAreaView>
            );
        }};

        const styles = StyleSheet.create({{
            container: {{ 
                flex: 1,
                // Add more styles 
            }},
            image: {{
                width: Dimensions.get('window').width * (100 / 375),
                height: Dimensions.get('window').width * (100 / 375),
            }},
        }});

        export default {screen_name}; 
        
        IMPLEMENTATION PATTERNS TO FOLLOW:
        - PROPER PASSWORD FIELD WITH VISIBILITY TOGGLE:
            ```jsx
            const [password, setPassword] = useState('');
            const [passwordVisible, setPasswordVisible] = useState(false);

            <View style={{styles.passwordContainer}}>
                <TextInput
                    style={{styles.passwordInput}}
                    value={{password}}
                    onChangeText={{setPassword}}
                    secureTextEntry={{!passwordVisible}}
                    placeholder="Password"
                    placeholderTextColor="#999"
                    accessibilityLabel="Password input"
                />
                <TouchableOpacity 
                    style={{styles.visibilityToggle}}
                    onPress={{() => setPasswordVisible(!passwordVisible)}}
                    accessibilityLabel="Toggle password visibility">
                    <Image 
                        source={{require('../assets/images/eye.png')}} 
                        style={{styles.visibilityIcon}} 
                    />
                </TouchableOpacity>
            </View>

            // Related styles:
            passwordContainer: {{
                position: 'relative',
                width: '100%',
                marginVertical: 10,
            }},
            passwordInput: {{
                width: '100%',
                borderWidth: 1,
                borderColor: '#ccc',
                borderRadius: 5,
                padding: 15,
                paddingRight: 50,
            }},
            visibilityToggle: {{
                position: 'absolute',
                right: 15,
                top: '50%',
                transform: [{{ translateY: -12 }}],
            }},
            visibilityIcon: {{
                width: 24,
                height: 24,
            }},
            ```
        
        
        - PROPER IMAGE COMPONENT:
            ```jsx
            <Image
                source={{require('../assets/images/image1.jpg')}}  // Use actual asset from Available Assets
                style={{styles.logoImage}}
                resizeMode="contain"
                accessibilityLabel="Logo"
            />
            ```
        
        
        - PROPER TEXT ALIGNMENT AND STYLING:
            ```jsx
            <Text style={{styles.centeredTitle}}>Welcome</Text>

            // Style:
            centeredTitle: {{
                fontSize: 24,
                fontWeight: 'bold',
                textAlign: 'center',
                width: '100%',
                marginVertical: 20,
                color: '#333',
            }},
            ```
        
        
        - PROPER "OR" DIVIDER WITH LINE:
            ```jsx
            <View style={{styles.dividerContainer}}>
                <View style={{styles.dividerLine}} />
                <Text style={{styles.dividerText}}>OR</Text>
                <View style={{styles.dividerLine}} />
            </View>

            // Styles:
            dividerContainer: {{
                flexDirection: 'row',
                alignItems: 'center',
                width: '100%',
                marginVertical: 20,
            }},
            dividerLine: {{
                flex: 1,
                height: 1,
                backgroundColor: '#E0E0E0',
            }},
            dividerText: {{
                paddingHorizontal: 10,
                color: '#888',
                fontSize: 14,
            }},
            ```
        
        
        - PROPER SOCIAL LOGIN BUTTON:
            ```jsx
            <TouchableOpacity 
                style={{styles.socialButton}}
                onPress={{() => console.log('Google login pressed')}}
                accessibilityLabel="Sign in with Google">
                <Image 
                    source={{require('../assets/images/google_logo.jpg')}} 
                    style={{styles.socialIcon}} 
                />
                <Text style={{styles.socialButtonText}}>Continue with Google</Text>
            </TouchableOpacity>

            // Styles:
            socialButton: {{
                flexDirection: 'row',
                alignItems: 'center',
                justifyContent: 'center',
                borderWidth: 1,
                borderColor: '#ddd',
                borderRadius: 5,
                padding: 12,
                marginVertical: 10,
                backgroundColor: '#fff',
            }},
            socialIcon: {{
                width: 24,
                height: 24,
                marginRight: 10,
            }},
            socialButtonText: {{
                fontSize: 16,
                color: '#333',
            }},
            ```
            Use this format exactly. Don't include explanations. Below is the analysis:
            {analysis}
            
            Note:
            - Provided code should be decent in styling; app widgets are to be clear and not overlapping when viewing the result.
            - Styling must be specified exactly as in the analysis and written to a high standard.
            - Styling should be a decent replica of the app page.
            - Password input fields with toggles must use a container View with the TextInput and toggle button as siblings, NOT as children of TextInput.
            - Every Image component must use a source from the Available Assets list (e.g., require('../assets/images/image1.jpg')).
            - Components with state like password visibility toggles must use useState.
            - Ensure no overlapping elements by using proper margins, padding, and flexbox layouts.
            - Use pixel-to-dp conversion (e.g., multiply pixel values by (Dimensions.get('window').width / 375)) for responsive scaling.
            - Import Dimensions from 'react-native' for pixel-to-dp conversion.
            """
        try:
            response = self.client.chat.completions.create(
                model=self.model_llm,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=8192,
                top_p=1,
                stream=False
            )
            content = response.choices[0].message.content
            start = content.find(f"{language_type}") + len(f"{language_type}") + 1
            end = content.rfind("```")
            code = content[start:end].strip()
            project_folder = os.path.join(self.project_base_dir, self.project_name or "temp")
            os.makedirs(project_folder, exist_ok=True)
            with open(os.path.join(project_folder, f"debug_raw_{screen_name}.log"), "w", encoding="utf-8") as f:
                f.write(f"Raw LLM Output:\n{content}")
            return self.validate_and_fix_code(code, language_type)
        except Exception as e:
            st.error(f"Code generation failed for {screen_name}: {str(e)}")
            return ""

    def generate_navigation_code(self, screens: List[str], language_type: str) -> str:
        """Generate navigation code for the app using React Navigation."""
        extension = "tsx" if language_type == "typescript" else "jsx"
        screen_imports = "\n".join(f"import {screen} from '../screens/{screen}';" for screen in screens)
        initial_screen = screens[0] if screens else "Home"
        screen_components = "\n      ".join(
            f"<Stack.Screen name=\"{screen}\" component={{{screen}}} />" for screen in screens
        )
        prompt = f"""
        Generate a React Native navigation file in {language_type} for screens: {', '.join(screens)}.
        Use createStackNavigator from '@react-navigation/stack'.
        Import screens from '../screens/{{screen}}'.
        Set initialRouteName to '{initial_screen}'.
        Export as default 'AppNavigator'.
        Save as a .{extension} file in 'src/navigation'.
        Use the following import statements exactly:
        ```{language_type}
        import React from 'react';
        import {{ createStackNavigator }} from '@react-navigation/stack';
        {screen_imports}
        ```
        Return the code in this format:
        ```{language_type}
        {{code}}
    ```
    """
        try:
            response = self.client.chat.completions.create(
                model=self.model_llm,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=8000,
                top_p=1,
                stream=False
            )
            content = response.choices[0].message.content
            start = content.find(language_type) + len(language_type) + 3
            end = content.rfind("```")
            return content[start:end].strip()
        except Exception as e:
            st.error(f"Navigation code generation failed: {str(e)}")
            return ""

    def setup_project_structure(self, project_name: str, framework: str, language_type: str, package_manager: str) -> Dict:
        """Set up the project directory structure and configuration files."""
        self.project_name = project_name
        project_folder = os.path.join(self.project_base_dir, project_name)
        structure = {
            "name": project_name,
            "framework": framework,
            "language": language_type,
            "packageManager": package_manager,
            "screens": [],
            "assets": [],
            "dependencies": self.determine_dependencies(framework)
        }
        os.makedirs(project_folder, exist_ok=True)
        os.makedirs(os.path.join(project_folder, "src/screens"), exist_ok=True)
        os.makedirs(os.path.join(project_folder, "src/assets/frames"), exist_ok=True)
        os.makedirs(os.path.join(project_folder, "src/assets/images"), exist_ok=True)
        os.makedirs(os.path.join(project_folder, "src/navigation"), exist_ok=True)
        self.generate_package_json(project_folder, framework, language_type, package_manager)
        self.generate_app_file(project_folder, language_type)
        return structure

    def determine_dependencies(self, framework: str) -> Dict:
        """Determine project dependencies based on the framework."""
        common_deps = {
            "react": "18.2.0",
            "react-native": "0.72.4",
            "@react-navigation/native": "^6.1.7",
            "@react-navigation/stack": "^6.3.17",
            "react-native-safe-area-context": "4.6.3",
        }
        if framework == "expo":
            common_deps.update({"expo": "~49.0.8"})
        return common_deps
    
    def generate_package_json(self, project_folder: str, framework: str, language_type: str, package_manager: str):
        """Generate package.json for the project."""
        dependencies = self.determine_dependencies(framework)
        package_json = {
            "name": self.project_name.lower().replace(" ", "-"),
            "version": "1.0.0",
            "main": "node_modules/expo/AppEntry.js" if framework == "expo" else "index.js",
            "scripts": {
                "start": "expo start" if framework == "expo" else "react-native start",
                "android": "expo start --android" if framework == "expo" else "react-native run-android",
                "ios": "expo start --ios" if framework == "expo" else "react-native run-ios"
            },
            "dependencies": dependencies,
            "devDependencies": {
                "@babel/core": "^7.20.0",
                **({
                    "@types/react": "~18.2.14",
                    "@types/react-native": "~0.72.2",
                    "typescript": "^5.1.3"
                } if language_type == "typescript" else {})
            }
        }
        with open(os.path.join(project_folder, "package.json"), "w", encoding="utf-8") as f:
            json.dump(package_json, f, indent=2)

    def generate_app_file(self, project_folder: str, language_type: str):
        """Generate the main App component."""
        extension = "tsx" if language_type == "typescript" else "jsx"
        app_code = f"""
            import React from 'react';
            import {{ NavigationContainer }} from '@react-navigation/native';
            import AppNavigator from './src/navigation/AppNavigator';

            const App{' : React.FC' if language_type == 'typescript' else ''} = () => {{
                return (
                    <NavigationContainer>
                        <AppNavigator />
                    </NavigationContainer>
                );
            }};

            export default App;
        """
        with open(os.path.join(project_folder, f"App.{extension}"), "w", encoding="utf-8") as f:
            f.write(app_code)

    def save_generated_code(self, code: str, screen_name: str, language_type: str):
        """Save generated code to the screens folder."""
        extension = "tsx" if language_type == "typescript" else "jsx"
        project_folder = os.path.join(self.project_base_dir, self.project_name)
        screens_folder = os.path.join(project_folder, "src/screens")
        with open(os.path.join(screens_folder, f"{screen_name}.{extension}"), "w", encoding="utf-8") as f:
            f.write(code)
        with open(os.path.join(project_folder, f"debug_{screen_name}.log"), "w", encoding="utf-8") as f:
            f.write(f"Generated Code:\n{code}")

    def save_navigation_code(self, code: str, language_type: str):
        """Save navigation code to the navigation folder."""
        extension = "tsx" if language_type == "typescript" else "jsx"
        project_folder = os.path.join(self.project_base_dir, self.project_name)
        nav_folder = os.path.join(project_folder, "src/navigation")
        with open(os.path.join(nav_folder, f"AppNavigator.{extension}"), "w", encoding="utf-8") as f:
            f.write(code)

    def generate_project_metadata(self, structure: Dict, screens: List[str]):
        """Generate project metadata file."""
        project_folder = os.path.join(self.project_base_dir, self.project_name)
        metadata = {
            "projectName": self.project_name,
            "generatedAt": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "screens": screens,
            "structure": structure,
            "assets": list(self.asset_map.values())
        }
        with open(os.path.join(project_folder, "project-metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
    
    def generate_code(self, project_name: str, file_key: str, framework: str = "react-native", 
                     language_type: str = "typescript", package_manager: str = "npm") -> Dict:
        """Generate React Native code for all Figma frames with actual image paths."""
        self.project_name = project_name
        self._image_counter = 0  # Reset counter for new project
        st.info("Fetching Figma frames...")
        try:
            frames = self.fetch_figma_frames(file_key)
            if not frames:
                st.error("No frames found. Aborting code generation.")
                return {"success": False, "screens": []}
            
            frame_ids = [frame["id"] for frame in frames]
            image_urls = self.fetch_figma_image_urls(file_key, frame_ids)
            image_node_urls = self.fetch_figma_image_urls(file_key, list(self.image_nodes.keys()))
            
            screens = []
            structure = self.setup_project_structure(project_name, framework, language_type, package_manager)
            extension = "tsx" if language_type == "typescript" else "jsx"

            # Download frame images
            for frame in frames:
                screen_name = frame["name"].replace(" ", "")
                screens.append(screen_name)
                st.info(f"Processing frame: {screen_name}")
                with open(os.path.join(self.project_base_dir, project_name, f"debug_frame_{screen_name}.json"), "w") as f:
                    json.dump(frame, f, indent=2)
                image_url = image_urls.get(frame["id"])
                try:
                    if image_url:
                        # Save frame image to images folder with generic name (e.g., image1.jpg)
                        frame_asset_name = f"image{self._image_counter + 1}.jpg"
                        self.asset_map[frame["id"]] = self.download_image(image_url, frame_asset_name, subfolder="images", is_frame=True)
                        # Save a copy to frames folder with screen_name (e.g., ConnectCCTV.jpg)
                        self.download_image(image_url, f"{screen_name}.jpg", subfolder="frames")
                        # Download image nodes for this frame
                        frame_image_nodes = {k: v for k, v in self.image_nodes.items() if v.get("frame_id") == frame["id"]}
                        for node_id, node_data in frame_image_nodes.items():
                            node_image_url = image_node_urls.get(node_id)
                            if node_image_url:
                                asset_name = f"{node_data['name']}.jpg"
                                self.asset_map[node_id] = self.download_image(node_image_url, asset_name, subfolder="images")
                        
                        analysis = self.analyze_figma_frame(image_url, frame)
                        if analysis:
                            code = self.generate_react_native_code(analysis, screen_name, language_type)
                            if code:
                                self.save_generated_code(code, screen_name, language_type)
                                st.success(f"✅ Generated code for {screen_name}")
                            else:
                                st.warning(f"Failed to generate code for {screen_name}")
                        else:
                            st.warning(f"Failed to analyze frame {screen_name}")
                    else:
                        st.warning(f"No image URL for frame {screen_name}")
                except Exception as e:
                    import traceback
                    st.error(f"Error processing frame {screen_name}: {str(e)}")
                    with open(os.path.join(self.project_base_dir, project_name, f"debug_error_{screen_name}.log"), "w") as f:
                        f.write(traceback.format_exc())
                    continue

            if screens:
                try:
                    nav_code = self.generate_navigation_code(screens, language_type)
                    self.save_navigation_code(nav_code, language_type)
                    st.success("✅ Generated navigation code")
                except Exception as e:
                    st.error(f"Failed to generate navigation code: {str(e)}")
                self.generate_project_metadata(structure, screens)

            success = len(screens) > 0 and any(os.path.exists(os.path.join(self.project_base_dir, project_name, "src/screens", f"{screen}.{extension}")) for screen in screens)
            if not success:
                st.error("No screen components were generated successfully.")
            return {"success": success, "project_name": project_name, "screens": screens, "structure": structure}
        except Exception as e:
            st.error(f"Unexpected error during code generation: {str(e)}")
            return {"success": False, "screens": []}