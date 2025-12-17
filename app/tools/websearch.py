import os
from typing import Optional, List, Dict, Any
from langchain_tavily import TavilySearch
from langchain_core.tools import tool
from dotenv import load_dotenv


load_dotenv()

# Initialize Tavily Search tool with LangChain
tavily_tool = TavilySearch(
    max_results=5,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=False,
    include_images=False,
    tavily_api_key=os.environ.get("TAVILY_API_KEY")
)


@tool
def web_search_nutrition(query: str) -> str:
    """
    Search the web for nutrition and food quality information using Tavily.

    Use this tool when you need to:
    - Verify nutrition facts and calorie information
    - Look up latest dietary guidelines, RDA (Recommended Dietary Allowance), or UL (Upper Limit)
    - Find brand or restaurant specific nutrition data
    - Check scientific claims about food and nutrition
    - Research food safety information
    - Find ingredient information or food additives

    Args:
        query: The search query about nutrition, food quality, or dietary information

    Returns:
        A formatted string containing search results with titles, URLs, and content snippets
    """
    try:
        results = tavily_tool.invoke({"query": query})

        if not results:
            return "No results found for the query."

        # Format results for better readability
        formatted_output = f"Search Results for: '{query}'\n\n"

        for idx, result in enumerate(results, 1):
            formatted_output += f"{idx}. {result.get('title', 'No title')}\n"
            formatted_output += f"   URL: {result.get('url', 'No URL')}\n"
            formatted_output += f"   Content: {result.get('content', 'No content')}\n"
            if result.get('score'):
                formatted_output += f"   Relevance Score: {result.get('score')}\n"
            formatted_output += "\n"

        return formatted_output

    except Exception as e:
        return f"Error performing web search: {str(e)}"


def get_tavily_search_tool(max_results: int = 5) -> TavilySearch:
    """
    Get a configured Tavily search tool instance for use with LangChain agents.

    Args:
        max_results: Maximum number of search results to return

    Returns:
        Configured TavilySearchResults tool
    """
    return TavilySearch(
        max_results=max_results,
        search_depth="advanced",
        include_answer=True,
        include_raw_content=False,
        include_images=False,
        tavily_api_key=os.environ.get("TAVILY_API_KEY")
    )
