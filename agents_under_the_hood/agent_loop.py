from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage, ToolMessage
from langchain.tools import tool
from langsmith import traceable

load_dotenv()

MODEL = "gemini-3.1-flash-lite"
MAX_ITERATION = 10


@tool
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog."""
    print(f"    >> Executing get_product_price(product='{product}')")
    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}
    return prices.get(product, 0)


@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return the final price.
    Available tiers: bronze, silver, gold."""
    print(
        f"    >> Executing apply_discount(price={price}, discount_tier='{discount_tier}')"
    )
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)


@traceable(name="ReAct agent model")
def run_agent(query: str):
    tools = [get_product_price, apply_discount]
    tools_dict = {t.name: t for t in tools}

    llm = init_chat_model(f"google_genai:{MODEL}")
    llm_with_tools = llm.bind_tools(tools)

    messages = [
        SystemMessage(
            content=(
                "You are a shoping assistent"
                "Use the tools to provide the prices of the products"
            )
        ),
        HumanMessage(content=query),
    ]

    for iteration in range(1, MAX_ITERATION + 1):
        print(f"Iteration-{iteration}")

        ai_message = llm_with_tools.invoke(messages)
        tool_calls = ai_message.tool_calls

        if not tool_calls:
            print(f"Final Answer: {ai_message.content}")
            return ai_message.content

        tool_name = tool_calls[0]["name"]
        tool_args = tool_calls[0]["args"]
        tool_id = tool_calls[0]["id"]

        print(f"  [Tool Selected] {tool_name} with args: {tool_args}")

        tool_called = tools_dict.get(tool_name)

        if tool_called is None:
            raise ValueError(f"Tool '{tool_name}' not found")

        observation = tool_called.invoke(tool_args)

        print(f"  [Tool Result] {observation}")

        messages.append(ai_message)
        messages.append(ToolMessage(content=str(observation), tool_call_id=tool_id))
    print("ERROR: Max iterations reached without a final answer")
    return None


if __name__ == "__main__":
    print("Hello LangChain Agent (.bind_tools)!")
    print()
    res = run_agent("what is the price of laptop after gold tier discount")
    res = run_agent("what is the price of laptop after gold tier discount")
