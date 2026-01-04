from langchain.agents import create_agent
from fastapi import FastAPI, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from langchain.tools import tool, ToolRuntime
from langgraph.checkpoint.memory import InMemorySaver
import uvicorn
from pydantic import BaseModel
    

import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Create shared checkpointer instance
checkpointer = InMemorySaver()

@tool
def add(a: int, b: int) -> int:
    '''Adds two numbers together.'''
    print(f"--------Adding {a} and {b}--------")
    return a + b

@tool
def multiply(a: int, b: int, runtime: ToolRuntime ) -> int:
    '''Multiplies two numbers together.'''
    print(f"--------Multiplying {a} and {b}--------")
    print(f"------- Runtime access in Tool - {runtime.context}")
    #print(f"------- Messages in State - {runtime.state['messages']}")
    return a * b

def get_agent():
    agent = create_agent(
        model="gpt-4o",
        tools=[add, multiply],
        checkpointer=checkpointer,
    )
    return agent



class AgentRequest(BaseModel):
    question: str
    model_call_count: int = 0
    continue_flag: str = "false"


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "fastapi-langchain-agent"}

@app.post("/run-agent")
@app.post("/run-agent/")
async def run_agent_endpoint(request: AgentRequest, user_id: str = Header(...)):
    agent = get_agent()
    result = agent.invoke(
        {"messages": request.question, "model_call_count": request.model_call_count},
        {"configurable": {"thread_id": user_id}},
        context={"user_id": user_id, "continue": request.continue_flag}
    )
    
    # Return only the last message content if available
    if isinstance(result, dict) and "messages" in result:
        return {"result": result["messages"][-1].content}
    return {"result": result}

## Not required for Lambda deployment
'''if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)'''
