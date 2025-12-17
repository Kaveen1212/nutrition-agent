import uvicorn
import os
import logging
import getpass
import platform
import hashlib
import tempfile
import traceback
import datetime
import json
import shutil
from fastapi import FastAPI, HTTPException, UploadFile, File
from PIL import Image
from fastapi.middleware.cors import CORSMiddleware
from app.agent.workflow import build_workflow
from pydantic import BaseModel, model_validator
from typing import Optional, List
from langchain_core.messages import HumanMessage, AIMessage

main = FastAPI(title="Food Quality Checking Agent API")

main.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

workflow_app = build_workflow()

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: Optional[str] = None
    messages: Optional[List[str]] = None
    history: Optional[List[ChatMessage]] = None

    @model_validator(mode='after')
    def ensure_message(self):
        """Make sure we always have some content to process."""
        if not self.message and not self.messages:
            raise ValueError("Either 'message' or 'messages' must be provided.")
        return self
    
    def combined_message(self) -> str:
        """Return a single markdown string combining message(s)."""
        if self.message:
            return self.message
        elif self.messages:
            return "\n\n".join(self.messages)
        return ""
    
class ChatResponse(BaseModel):
    role: str = "assistant"
    content: str

class UploadResponse(BaseModel):
    status: str
    detail: str
    document_path: Optional[str] = None

def get_user_id() -> str:
    """Generate a consistent user ID based on system information."""
    username = getpass.getuser()
    machine_name = platform.node()
    user_info = f"{username}@{machine_name}"
    user_hash = hashlib.md5(user_info.encode()).hexdigest()[:8]
    return f"user_{user_hash}"


@main.get("/health")
def health():
    return {"status": "ok"}

# chat Endpoint 
@main.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
        config = {
            "configurable":{
                 "thread_id": f"{get_user_id()}_api_session",
                "user_id": get_user_id(),
            }
        }

        combined_text = req.combined_message()

        messages = []
        if req.history:
            for msg in req.history:
                if msg.role == "user":
                    messages.append(HumanMessage(content=msg.content))
                else:
                    messages.append(AIMessage(content=msg.content))
        messages.append(HumanMessage(content=combined_text))

        input_state = {"messages": messages}

        response_content = ""
        for step in workflow_app.stream(input_state, config, stream_mode="values"):
            messages = step["messages"][-1]
            if hasattr(messages, "content"):
                response_content = messages.content

        if not response_content:
            response_content = (
                 "I apologize, but I couldn't generate a response. "
                "Please try rephrasing your question."
            )

        return ChatResponse(content=response_content)

    except ImportError as e:
        return ChatResponse(content=f"**Configuration Error:** {e}")
    except Exception as e:
        return ChatResponse(content=f"**Error:** {e}")
    
# Image uploading endpoint 
@main.post("/api/upload", response_model=UploadResponse)
async def upload_image (file: UploadFile = File(...), description: str = ""):
    """Upload meals image, save it to uploads/meals, and analyze using OpenAI Vision API."""

    allowed_extensions = (".png", ".jpg", ".jpeg")
    if not file.filename.lower().endswith(allowed_extensions):
        raise HTTPException(status_code=400, detail="Only PNG, JPG, and JPEG images are accepted.")
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            tmp_path = tmp.name
            contents = await file.read()
            tmp.write(contents)
    except Exception as e:
        logger.error("Failed to save image upload: %s\n%s", e, traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to save upload: {e}")
    
    # Validate image 
    def validate_image(path: str):
        try:
            with Image.open(path) as img:
                img.verify()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

    try:
        validate_image(tmp_path)
    except HTTPException:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        raise

    # meals.json 
    meals_json_path = os.path.join("app", "tools", "meals.json")
    uploads_dir = os.path.join("uploads", "meals")
    os.makedirs(uploads_dir, exist_ok=True)

    try:
        if os.path.exists(meals_json_path):
            with open(meals_json_path, 'r', encoding='utf-8') as f:
                meals_data = json.load(f)
        else:
            meals_data = []

        new_id = max([meal.get("id", 0) for meal in meals_data], default=0) + 1

        file_extension = os.path.splitext(file.filename)[1]
        new_filename = f"meals_{new_id}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}{file_extension}"
        final_image_path = os.path.join(uploads_dir, new_filename)
        shutil.move(tmp_path, final_image_path)

        # Create plant record
        new_meals = {
            "id": new_id,
            "name": file.filename,
            "description": description or "No description provided",
            "image_path": final_image_path,
            "upload_date": datetime.datetime.now().isoformat(),
            "user_id": get_user_id()
        }

        analysis_result = None 

         # Analyze with OpenAI Vision API
        try:
            from app.tools.meals_detect import analyze_meal_image
            
            question = f"{description} with this image" if description else "Analyze this meal image"
            
            analysis = analyze_meal_image(final_image_path, question)
            
            new_meals["analysis_response"] = analysis
            analysis_result = analysis
            logger.info(f"OpenAI Vision analysis completed for meals {new_id}")
        except Exception as e:
            logger.warning(f"OpenAI Vision analysis failed for meals {new_id}: {e}\n{traceback.format_exc()}")
            error_result = {"error": str(e)}
            new_meals["analysis_response"] = error_result
            analysis_result = error_result

        meals_data.append(new_meals)
        with open(meals_json_path, 'w', encoding='utf-8') as f:
            json.dump(meals_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Successfully uploaded and analyzed meal {new_id}: {new_filename}")
        
        # Return success response
        return UploadResponse(
            status="success",
            detail=f"Meal image uploaded and analyzed successfully. ID: {new_id}",
            document_path=final_image_path,
            analysis=analysis_result 
        )

    except Exception as e:
        logger.error("Failed to save meal data: %s\n%s", e, traceback.format_exc())
        try:
            if 'final_image_path' in locals():
                os.remove(final_image_path)
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"Failed to save meal data: {e}")

if __name__ == "__main__":
    uvicorn.run("main:main", host="0.0.0.0", port=8000, reload=True)