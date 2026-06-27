from llm_runtime import (
    Agent,
    DeepSeekTokenCounter,
    LLMConfig,
    SummarizeCompressor,
    Tool,
    register_basic_linux_tools,
)
import os
import asyncio
from dotenv import load_dotenv

load_dotenv(override=True)


def _build_compressor() -> SummarizeCompressor | None:
    """Create a compressor from env vars, or None if LLM_MAX_CONTEXTS is unset."""
    raw = os.getenv("LLM_MAX_CONTEXTS")
    if raw is None:
        return None
    try:
        max_tokens = int(raw)
    except ValueError:
        return None
    if max_tokens <= 0:
        return None
    counter = DeepSeekTokenCounter()
    return SummarizeCompressor(counter, max_context_tokens=max_tokens)


def get_agent(llm_config: LLMConfig, debug: bool = False) -> Agent:
    compressor = _build_compressor()
    agent: Agent = Agent(llm_config, debug=debug, compressor=compressor)
    register_basic_linux_tools(agent)
    return agent

async def chat_with_human(agent:Agent,system_prompt:str)-> None:
    observations: list[dict] = [
        {
            "role":"system",
            "content":system_prompt,
        }
    ]
    while True:
        user_prompt = input("<long horizon coding agent>:")
        observations.append({
            "role":"user",
            "content":user_prompt,
        })
        observations = await agent.chat(observations)

async def loop_engineering_static(agent:Agent,system_prompt:str,goal:str,max_rounds: int = 50)-> None:
    round: int = 0
    while round < max_rounds:
        observations: list[dict] = [
            {
                "role":"system",
                "content":system_prompt,
            }
        ]
        round += 1
        user_prompt = goal
        observations.append({
            "role":"user",
            "content":user_prompt,
        })
        observations = await agent.chat(observations)

async def loop_engineering_dynamic(agent:Agent,system_prompt:str,critic_agent:Agent,critic_system_prompt:str,goal:str)-> None:
    # todo
    observations: list[dict] = [
        {
            "role":"system",
            "content":system_prompt,
        }
    ]
    critic_observations: list[dict] = [
        {
            "role":"system",
            "content":critic_system_prompt,
        }
    ]
    feedback = None
    while True:
        user_prompt = goal
        if feedback is not None:
            user_prompt = "original goal:" + goal + "\n" + "now the feedback of critic agent:" + feedback
        observations.append({
            "role":"user",
            "content":user_prompt,
        })
        observations = await agent.chat(observations)
    


if __name__ == "__main__":
    llm_config = LLMConfig(
        api_key = str(os.getenv("LLM_API_KEY")),
        model = str(os.getenv("LLM_MODEL")),
        base_url = str(os.getenv("LLM_BASE_URL")),
    )
    agent = get_agent(llm_config)
    # asyncio.run(chat_with_human(agent,"you are a coding assistant,you can use tools to help user solve the coding problem."))
    # asyncio.run(loop_engineering_static(agent,"you are a coding assistant,you can use tools to help user solve the coding problem.",
    # goal="帮我完成README.md实验，我们在wsl ubuntu24上，内存16GB，swap 4GB，我们已经写了一些代码，我们有虚拟环境，里面安装好了vllm和torch,请你不要改动我们的虚拟环境，我们也有qwen3.5-2B,请你帮我把README.md里的实验code后跑出来，如果遇到问题一直迭代完成这个goal"))
    asyncio.run(loop_engineering_static(agent,"you are a coding assistant,you can use tools to help user solve the coding problem.",
    goal="我们有/home/zane/ReactS/deepseek_tokenizer，我们用/home/zane/ReactS/.venv虚拟环境，我们用uv管理python依赖，已经安装了一些包了比如vllm,pytorch,你千万不要删除这个.venv，要不然我们重新装.venv我会崩溃！记住：你可以uv add或者uv pip加东西，但是不要删除.venv!这是死律！你只管把代码优化到极致（根据我们的工程观），不用动git先，我们现在代码里observations的记忆压缩机制还不完整，我的工程观你一定要满足：编写大型项目的时候，正确遵守设计原则就变得非常必要。开闭原则在5w行以内重要性还没有那么凸显出来，但随着项目体积的累计，尽量遵守开闭原则几乎成了repo的生命线。开闭原则说的是，对修改关闭，对拓展开放。一方面本身SE就遵守项目越大，已经被验证work已经被test的代码就不要再轻易加参数穿插路径去改动了，因为心智负担极大，改动成本极高。vibe coding时代更是如此。很多人说AI今天写代码很厉害了，但是项目庞大起来，AI的上下文有限，在庞大的repo中穿插各种新功能的路径和参数，代码迭代和维护的心智负担会超级超级大。如果能尽量维护开闭原则，一方面对AI上下文的注意力集中有好处（大部分精力专心于产出连贯的新代码），一方面本身也符合SE的本质特点。大型项目，解耦做依赖注入同样非常必要，我觉得一个很好的规范是，假如你是vibe coder，对自己repo的一个约束可以是：每个文件最多250行，考虑到跨文件可能会造成反向的可读性缺失，允许你最多3个文件可以超过250行但是仍然不能超过600行（但是尽量还是都250行以内）。可读性极强，解耦优秀，单测集测量化覆盖率大于95%。如果设计的时候能按照这样的约束来强迫自己，那么设计出的系统那概率是能经得起迭代和维护的。用文件体积倒逼需求和设计的解耦，用高测试率快速锁定大部分已经写好的逻辑，是个很好的方式。所有coder平常写代码必须用git，必须做version control，这是死律。if you write code without using git,don't say you are a coder.我认为正确使用git是工程能力提高的必要条件。我建议找几个小伙伴，找一个空闲的下午，大家边喝咖啡，边一起实操探索git的各种玩法，it only takes half of a day to master git，using this way.此外，我还观察到infra或者开发大佬的github无一例外全年一片绿色，这也是非常重要的。今天很多人做vibe coder，产出了很多代码，但是没有review就没有进步，没有test就没有质量，wrong plan wrong design，no doc-code sync，no review，no test四者但凡只要占一样，repo都活不过代码行数5w+的那一天。以上是一些如何做如何迭代如何维护大型项目的心得。"))