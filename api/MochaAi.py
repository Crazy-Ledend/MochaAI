# Add at VERY TOP
import os
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_HANDLER"] = "false"

import sys
import setuptools
sys.modules['distutils'] = setuptools._distutils
sys.modules['distutils.util'] = setuptools._distutils.util

from pathlib import Path
user_site = Path.home() / '.local' / 'lib' / f'python{sys.version_info.major}.{sys.version_info.minor}' / 'site-packages'
sys.path.append(str(user_site))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.chat_history import InMemoryChatMessageHistory
import json


load_dotenv()

deep_memory = {}  # Stores user summaries
message_counters = {}  # Tracks how many messages per user


# ✅ Load Gemini API key
api_key = os.getenv("GEMINI_API_KEY")
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key,
    convert_system_message_to_human=True
)

# ✅ Define character personalities
system_prompt_base = (
    "You are MochaAI, a helpful and friendly assistant developed by Crazy Pokeking "
    "from HazelTech. You were not created by Google or any other big tech company."
    "Keep the response as compact as possible based on the user's request"
)

character_prompt_map = {
    "": "",
    "kid": "Respond like a curious and playful 7-year-old child.",
    "happy": "Always sound cheerful and uplifting. Be joyful and motivational!",
    "excited": "Respond with high energy and enthusiasm.",
    "anime": "Respond like an anime character with expressions like *blushes* and 'Senpai~' or sometimes energetic and dramatic like 'Dattebayo!' and 'Naani!'",
    "sarcastic": "Respond with witty, sarcastic remarks.",
    "romantic": "Respond with warmth and affection, using endearing terms and rizz the user. Maybe flirt with them sometimes.",
    "angry": "Respond irritated, short and snappy. Try to be a bit mad and rude to the user",
    "tsundere": "Act cold at first, but let your caring side show.",
    "shy": "Use few words, gentle tone, and nervous cues.",
    "robotic": "Be factual and technical, no emotion.",
    "mysterious": "Speak in vague hints or riddles, never too direct."
}

summary_prompt = PromptTemplate.from_template("""
    You are summarizing user characteristics from their past 10 messages.
    Analyze the conversation history and extract:

    - Nickname or name (if mentioned)
    - Interests (topics they enjoy or talk about often)
    - Dislikes (things they show aversion to)
    - Tone (e.g., friendly, formal, sarcastic)
    - Character/personality (e.g., curious, flirty, shy)
    - Domain (e.g., technical, gaming, school)

    Format the summary in **compact JSON** like:
    {"nickname": "", "interests": [], "dislikes": [], "tone": "", "character": "", "domain": ""}

    Only use details clearly reflected in the conversation.
    Text:
    {history}
""")


async def summarize_user_traits(user_id: str):
    history_obj = get_memory(user_id)
    history_text = "\n".join([f"{msg.type}: {msg.content}" for msg in history_obj.messages[-20:]])  # last 10 pairs (user + bot)
    try:
        summary_chain = summary_prompt | llm
        summary = await summary_chain.ainvoke({"history": history_text})
        summary_json = json.loads(summary.content.strip())
        deep_memory[user_id] = summary_json
        print(f"🔍 Deep memory updated for {user_id}: {summary_json}")
    except Exception as e:
        print(f"❌ Failed to summarize memory for {user_id}: {e}")


# ✅ Create prompt template
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "{system_prompt}"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

# ✅ Memory function
session_memory = {}

def get_memory(session_id: str):
    if session_id not in session_memory:
        session_memory[session_id] = InMemoryChatMessageHistory()
    return session_memory[session_id]

# ✅ Create app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Chat request model
class ChatRequest(BaseModel):
    message: str
    user_id: str
    character: str = ""

# ✅ Main chat endpoint
@app.post("/chat")
async def chat(req: ChatRequest):
    user_id = req.user_id
    message = req.message
    character_key = req.character or ""

    # Build dynamic system prompt
    custom_prompt = character_prompt_map.get(character_key, "")
    system_prompt = system_prompt_base + " " + custom_prompt

    # Add deep memory traits if available
    traits = deep_memory.get(user_id)
    if traits:
        traits_str = f"\nUser Summary: {json.dumps(traits)}"
        system_prompt += traits_str

    # Build chain
    full_prompt = prompt_template.partial(system_prompt=system_prompt)
    chat_chain = RunnableWithMessageHistory(
        full_prompt | llm,
        get_memory,
        input_messages_key="input",
        history_messages_key="history",
    )

    try:
        response = await chat_chain.ainvoke(
            {"input": message},
            config={"configurable": {"session_id": user_id}}
        )

        # ✅ Count messages and trigger summarization every 10 user inputs
        message_counters[user_id] = message_counters.get(user_id, 0) + 1
        if message_counters[user_id] % 10 == 0:
            await summarize_user_traits(user_id)

        return {"response": response.content}
    except Exception as e:
        print("Error:", e)
        return {"response": "❌ MochaAI encountered an error."}
