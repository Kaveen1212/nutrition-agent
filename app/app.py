from app.agent.workflow import build_workflow
from langchain_core.messages import HumanMessage

def app():
    
    workflow = build_workflow()
    
    config = {"configurable": {"thread_id": "1"}}
    
    user_input = "Analyze this meal: grilled chicken breast (150g), brown rice (1 cup), steamed broccoli (100g)"
    
    result = workflow.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        config=config
    )
    
    print("\n" + "="*80)
    print("AGENT RESPONSE:")
    print("="*80)
    print(result["messages"][-1].content)

if __name__ == "app":
    app()