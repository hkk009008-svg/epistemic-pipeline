import asyncio
import os
from pipeline.runner import generate_pipeline_async
from pipeline.models import PipelineRequest
import config

async def main():
    # Force the key for this local test script
    key = os.getenv("OPENAI_API_KEY")
    config.set_runtime_config(key, "gpt-4o-mini")
    
    long_query = "a" * 500
    req = PipelineRequest(prompt=long_query)
    resp = await generate_pipeline_async(req)
    print("Verdict:", resp.final_verdict)
    
    for hr in resp.hook_results:
        print(f"Hook: {hr.tool_name if hasattr(hr, 'tool_name') else hr.tool}")
        print(f"Note: {hr.note}")

asyncio.run(main())
