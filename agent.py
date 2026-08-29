import asyncio
import os
import httpx
from google import genai
from google.genai import types
from pydantic import BaseModel, Field, ValidationError

# 1. Pydantic request model matching server's QueryRequest
class QueryRequest(BaseModel):
    query: list[float]
    k: int = Field(..., ge=1, le=100)

# Final expected output structure from the agent
class FinalAnswer(BaseModel):
    summary: str
    indices: list[int]
    top_similarities: list[float]

# 2. Declare tool schema for Gemini API
search_tool = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="search_db",
            description="Performs vector similarity search against the corpus.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "query": types.Schema(
                        type="ARRAY",
                        items=types.Schema(type="NUMBER"),
                        description="Vector representation list of floats. Dimension must equal 4."
                    ),
                    "k": types.Schema(
                        type="INTEGER",
                        description="Number of top matching vector items to return (1 to 100)."
                    )
                },
                required=["query", "k"]
            )
        )
    ]
)

FASTAPI_URL = "http://localhost:8000/search_db"
MAX_RETRIES = 3

async def execute_search_tool(args: dict) -> dict:
    """Validates parameters with Pydantic and executes the endpoint call via httpx."""
    # Local Pydantic validation
    validated_args = QueryRequest(**args)

    # Async API call to FastAPI endpoint
    async with httpx.AsyncClient() as client:
        response = await client.post(FASTAPI_URL, json=validated_args.model_dump())
        
        if response.status_code != 200:
            raise ValueError(f"HTTP {response.status_code} Error: {response.text}")
            
        return response.json()


async def run_agent_loop(user_query: str):
    client = genai.Client()
    model_id = "gemini-2.5-flash"

    # Start a chat session with tool awareness
    chat = client.chats.create(
        model=model_id,
        config=types.GenerateContentConfig(
            tools=[search_tool],
            temperature=0.1
        )
    )

    response = chat.send_message(user_query)

    attempts = 0
    while attempts < MAX_RETRIES:
        # Check if the model requests a function call
        if response.function_calls:
            function_call = response.function_calls[0]
            func_name = function_call.name
            func_args = function_call.args

            print(f"[Attempt {attempts + 1}] Model invoked function: {func_name} with args: {func_args}")

            try:
                # Attempt tool validation and execution
                if func_name == "search_db":
                    tool_result = await execute_search_tool(func_args)
                    print("[Success] Tool returned:", tool_result)

                    # Send successful tool execution result back to Gemini
                    response = chat.send_message(
                        types.Part.from_function_response(
                            name=func_name,
                            response={"result": tool_result}
                        )
                    )
                    break

            except (ValidationError, ValueError) as err:
                attempts += 1
                error_msg = f"Tool execution failed with error: {str(err)}. Please correct your arguments and retry."
                print(f"[Error] {error_msg}")

                if attempts >= MAX_RETRIES:
                    raise RuntimeError(
                        f"Failed after {MAX_RETRIES} attempts. Hard cap reached. Last error: {str(err)}"
                    )

                # Feed error message back into the loop for self-correction
                response = chat.send_message(
                    types.Part.from_function_response(
                        name=func_name,
                        response={"error": error_msg}
                    )
                )
        else:
            # Model responded directly without calling a tool
            break

    # Prompt model for final structured response
    final_response = client.models.generate_content(
        model=model_id,
        contents=f"Summarize these findings: {response.text}",
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=FinalAnswer
        )
    )

    return final_response.text


if __name__ == "__main__":
    # Ensure GEMINI_API_KEY environment variable is set
    sample_prompt = "Find the top 2 vectors similar to query [0.5, 0.1, 0.9, 0.2]."
    
    try:
        result = asyncio.run(run_agent_loop(sample_prompt))
        print("\n--- Final Structured Answer ---")
        print(result)
    except Exception as e:
        print(f"\n[Agent Failure] {e}")
