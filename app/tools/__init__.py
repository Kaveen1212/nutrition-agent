from typing import List
from langchain_core.tools import BaseTool
from .websearch import web_search_nutrition

def get_tools(llm=None) -> List[BaseTool]:
    """Return a list of available tools for the agent."""
    return [web_search_nutrition]
__all__ = ["get_tools"]