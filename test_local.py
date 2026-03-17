import asyncio
import os
from pipeline.runner import generate_pipeline_async
from pipeline.models import PipelineRequest
import config

async def main():
    # Force the key for this local test script
    key = os.getenv("OPENAI_API_KEY")
    config.set_runtime_config(key, "gpt-4o-mini")
    
    req = PipelineRequest(prompt='Spinach is famously considered an exceptional source of iron because a German chemist in 1870 accidentally misplaced a decimal point')
    resp = await generate_pipeline_async(req)
    print("Verdict:", resp.final_verdict)
    print("Reason:", resp.reasoning)

asyncio.run(main())
