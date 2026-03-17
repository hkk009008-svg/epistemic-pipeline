from langgraph.graph import StateGraph, START, END

from pipeline.state import GraphState
from pipeline.nodes.init import stage_init
from pipeline.nodes.route import stage_route
from pipeline.nodes.hooks import stage_hooks
from pipeline.nodes.evaluate import stage_evaluate
from pipeline.nodes.build_response import stage_build_response


def build_pipeline_graph() -> StateGraph:
    """Constructs the Epistemic Pipeline as a LangGraph StateGraph."""
    
    workflow = StateGraph(GraphState)

    # 1. Add all nodes
    workflow.add_node("init", stage_init)
    workflow.add_node("route", stage_route)
    workflow.add_node("hooks", stage_hooks)
    workflow.add_node("evaluate", stage_evaluate)
    workflow.add_node("build_response", stage_build_response)

    # 2. Define sequential flow
    workflow.add_edge(START, "init")
    workflow.add_edge("init", "route")
    workflow.add_edge("route", "hooks")
    workflow.add_edge("hooks", "evaluate")
    workflow.add_edge("evaluate", "build_response")
    workflow.add_edge("build_response", END)

    return workflow.compile()
