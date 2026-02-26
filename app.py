from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.base import TaskResult
from dotenv import load_dotenv
import os 

# optional Google Gemini client wrapper
from google_client import GoogleGeminiClient

from typing import Optional
from pydantic import BaseModel
import base64
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


load_dotenv()

app = FastAPI()


# 1. Mount Static Files and Templates
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/output_images", StaticFiles(directory="output_images"), name="output_images")
templates = Jinja2Templates(directory="templates")



# load keys from environment; the code supports either OpenAI or Google
# Gemini.  If both are present, Google takes precedence.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if GOOGLE_API_KEY:
    # use the custom wrapper defined in google_client.py
    model_client = GoogleGeminiClient(api_key=GOOGLE_API_KEY)
elif OPENAI_API_KEY:
    model_client = OpenAIChatCompletionClient(model="gpt-4o", api_key=OPENAI_API_KEY)
else:
    raise RuntimeError("No API key provided; set OPENAI_API_KEY or GOOGLE_API_KEY")



# --- WebSocket Handler ---
class WebSocketInputHandler:
    def __init__(self, websocket: WebSocket):
        self.websocket = websocket

    async def get_input(self, prompt: str, cancellation_token: Optional[object] = None) -> str:
        try:
            # Signal frontend that it's user's turn
            await self.websocket.send_text("SYSTEM_TURN:USER")
            data = await self.websocket.receive_text()
            return data
        except WebSocketDisconnect:
            print("Client disconnected during input wait.")
            return "TERMINATE"
        


async def create_interview_team(websocket: WebSocket, job_position: str):
    handler = WebSocketInputHandler(websocket)

    interviewer = AssistantAgent(
        name="Interviewer",
        model_client=model_client,
        description=f"Interviewer for {job_position}",
        system_message=f'''
        You are a professional interviewer for a {job_position} position.
        Ask ONE question at a time and wait for the candidate to respond before continuing.
        Ask exactly 3 questions in this order: Technical, Problem Solving, Culture fit.
        Keep each question under 50 words.
        IMPORTANT: Only after the candidate has answered ALL 3 questions AND the evaluator has given feedback on the 3rd answer, send a final message that says ONLY the word: TERMINATE
        Do NOT combine a question and TERMINATE in the same message.
        '''
    )

    candidate = UserProxyAgent(
        name="Candidate",
        description="The candidate",
        input_func=handler.get_input 
    )

    evaluator = AssistantAgent(
        name="Evaluator",
        model_client=model_client,
        description="Career Coach",
        system_message=f'''
        You are a career coach. Give very brief feedback (max 40 words) on the candidate's answer.
        '''
    )

    terminate_condition = TextMentionTermination(text="TERMINATE")

    return RoundRobinGroupChat(
        participants=[interviewer, candidate, evaluator],
        termination_condition=terminate_condition,
        max_turns=20
    )


# --- Routes ---

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    # Render the index.html template
    return templates.TemplateResponse("index.html", {"request": request})


class ChatImageData(BaseModel):
    image_base64: str
    job_role: str

@app.post("/api/save-chat-screenshot")
async def save_chat_screenshot(data: ChatImageData):
    try:
        header, encoded = data.image_base64.split(",", 1)
        img_data = base64.b64decode(encoded)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_role = data.job_role.replace(" ", "_").replace("/", "")
        filename = f"{timestamp}_chat_{safe_role}.png"
        
        output_dir = os.path.join(os.path.dirname(__file__), "output_images")
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, "wb") as f:
            f.write(img_data)
            
        print(f"[screenshot] Saved chat UI to output_images/{filename}")
        return {"status": "success", "file": filename}
    except Exception as e:
        print(f"Error saving chat screenshot: {e}")
        return {"status": "error", "message": str(e)}


@app.websocket("/ws/interview")
async def websocket_endpoint(websocket: WebSocket, pos: str = Query("AI Engineer")):
    await websocket.accept()
    session_messages: list[dict] = []   # collect all messages for report
    try:
        team = await create_interview_team(websocket, pos)

        await websocket.send_text(f"SYSTEM_INFO:Starting interview for {pos}...")

        async for message in team.run_stream(task='Start the interview.'):
            if isinstance(message, TaskResult):
                await websocket.send_text(f"SYSTEM_END:{message.stop_reason}")
            else:
                session_messages.append({
                    "source": message.source,
                    "content": message.content,
                })
                await websocket.send_text(f"{message.source}:{message.content}")

    except WebSocketDisconnect:
        print("WebSocket disconnected.")
    except Exception as e:
        print(f"Error: {e}")