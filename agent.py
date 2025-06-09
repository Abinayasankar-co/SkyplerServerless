from langgraph.graph import StateGraph
from langchain_core.runnables import RunnableLambda
from tools.structure_generator import create_project_structure
from tools.file_writer import create_project_structure as write_code_to_files
from orchestrator import plan_tasks, analyze_codebase
from guider import run_guider
import threading

def planning_node(state):
    state["tasks"] = plan_tasks(state["project_description"], state["figma_access_token"] , state["file_key"] )
    return state

def structure_node(state):
    if state.get("tasks"):
        create_project_structure("generated_app", state["tasks"])
    return state

def write_code_node(state):
    if state.get("tasks"):
        write_code_to_files("generated_app", state["tasks"])
    return state

def analysis_node(state):
    issues = analyze_codebase("generated_app")
    state["issues"] = issues
    return state

def guider_node(state):
    run_guider(state)
    return state

def start_guider_thread(state):
    thread = threading.Thread(target=run_guider, args=(state,), daemon=True)
    thread.start()

def build_app():
    builder = StateGraph(input=planning_node, output=write_code_node)
    builder.add_node("PlanTasks", RunnableLambda(planning_node))
    builder.add_node("InitStructure", RunnableLambda(structure_node))
    builder.add_node("WriteCode", RunnableLambda(write_code_node))
    builder.add_node("Analyze", RunnableLambda(analysis_node))
    builder.add_node("Guider", RunnableLambda(guider_node))

    builder.set_entry_point("PlanTasks")
    builder.add_edge("PlanTasks", "InitStructure")
    builder.add_edge("InitStructure", "WriteCode")
    builder.add_edge("WriteCode", "Analyze")
    builder.add_edge("Analyze", "Guider")

    return builder.compile()

app = build_app()
