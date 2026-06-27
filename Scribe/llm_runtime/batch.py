try:
    # Package import path (llm_runtime.batch)
    from .tool_calling import Agent
except Exception:  # pragma: no cover - script mode fallback
    # Script-mode fallback (python llm_runtime/batch.py)
    from tool_calling import Agent

import asyncio

async def batch(agent: Agent, observations: list[list[dict]], max_concurrent: int = 20) -> list[list[dict]]:
    '''
    Call the LLM concurrently in batches with a maximum concurrency limit.
    
    A semaphore controls concurrency: a new task starts as soon as a slot is
    available, keeping up to max_concurrent tasks running until all tasks finish.
    
    Args:
        agent: Agent instance
        observations: Initial LLM context list; each context is a list[dict]
        max_concurrent: Maximum concurrency, defaulting to 20
    
    Returns:
        list[list[dict]]: Result list in the same order as observations
    '''
    semaphore = asyncio.Semaphore(max_concurrent)
    num = len(observations)

    assert num > 0, "observations cannot be empty"
    assert max_concurrent > 0, "max_concurrent must be at least 1"

    async def _execute_with_index(idx: int) -> tuple[int, list[dict]]:
        '''Execute a single task and return (index, result).'''
        async with semaphore:  # Acquire a slot, waiting if none is available
            result = await agent.chat(observations[idx])
            return idx, result
    
    # Create all tasks; the semaphore prevents them from starting all at once
    tasks = [_execute_with_index(i) for i in range(num)]
    
    # Collect all results
    results_with_index = await asyncio.gather(*tasks)
    
    # Sort by original index to preserve input order
    results_with_index.sort(key=lambda x: x[0])
    
    # Extract results
    results = [r[1] for r in results_with_index]
    return results
