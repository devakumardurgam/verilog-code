import os
import traceback
from typing import TypedDict, List, Optional

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI


# ============================================================
# CONFIGURATION
# ============================================================

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError(
        "GEMINI_API_KEY is not set. "
        "Set it as an environment variable before running the app."
    )

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite-preview",
    google_api_key=API_KEY,
    temperature=0.2,
)


# ============================================================
# STATE
# ============================================================

class CrewState(TypedDict):
    messages: List[BaseMessage]
    next_step: Optional[str]
    code: Optional[str]
    report: Optional[str]


# ============================================================
# HELPERS
# ============================================================

def response_to_text(response) -> str:
    content = getattr(response, "content", response)

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(item.get("text", ""))
            else:
                parts.append(str(item))
        return "\n".join(parts)

    return str(content)


# ============================================================
# VERILOG GENERATION TOOL
# ============================================================

@tool
def generate_verilog_code(task_description: str) -> str:
    """Generate synthesizable Verilog RTL for a digital-design task."""

    prompt = f"""
You are an expert RTL, Verilog and SystemVerilog engineer.

Generate Verilog RTL code for this digital design task:

{task_description}

Requirements:
1. Generate synthesizable RTL.
2. Prefer standard Verilog-2001 syntax unless SystemVerilog is explicitly requested.
3. Include a meaningful module name.
4. Include all required ports.
5. Correctly implement the requested functionality.
6. Use always blocks appropriately.
7. Use non-blocking assignments for sequential logic.
8. Include reset logic when required.
9. Return ONLY source code.
10. Do NOT generate Python.
11. Do NOT provide explanations.
12. Do NOT use Markdown code fences.
"""

    response = llm.invoke(prompt)
    code = response_to_text(response)

    code = (
        code.replace("```verilog", "")
        .replace("```systemverilog", "")
        .replace("```", "")
        .strip()
    )

    return code


# ============================================================
# TEST SCENARIO GENERATION TOOL
# ============================================================

@tool
def generate_verilog_test_cases(task_description: str) -> str:
    """Generate RTL verification scenarios for a digital-design task."""

    prompt = f"""
You are a Senior RTL Verification Engineer.

Generate 3 to 5 specific test scenarios for:

{task_description}

Include:
1. Normal cases
2. Boundary cases
3. Reset behavior, if applicable
4. Important input combinations
5. Edge cases

Return only a numbered list.
Do not generate Verilog code.
"""

    response = llm.invoke(prompt)
    return response_to_text(response).strip()


# ============================================================
# TASK INPUT NODE
# ============================================================

def task_input_node(state: CrewState):
    print("\n" + "=" * 60)
    print("        VERILOG AI DESIGN ASSISTANT")
    print("=" * 60)

    user_task = input(
        "\nEnter the Verilog design task "
        "(or type 'exit' to quit): "
    ).strip()

    if user_task.lower() == "exit":
        return {"next_step": "exit"}

    return {
        "messages": [HumanMessage(content=user_task)],
        "next_step": "developer",
    }


# ============================================================
# DEVELOPER NODE
# ============================================================

def real_time_verilog_developer(state: CrewState):
    print("\n[RTL DEVELOPER] Generating Verilog code...")

    task = state["messages"][-1].content

    try:
        code = generate_verilog_code.invoke(task)

        print("\n" + "-" * 60)
        print("GENERATED VERILOG")
        print("-" * 60)
        print(code)
        print("-" * 60)

        return {"code": code}

    except Exception:
        error_message = (
            "Verilog generation failed:\n"
            + traceback.format_exc()
        )
        print(error_message)
        return {"code": error_message}


# ============================================================
# TESTER NODE
# ============================================================

def real_time_verilog_tester(state: CrewState):
    print("\n[VERIFICATION ENGINEER] Generating test scenarios...")

    task = state["messages"][-1].content

    try:
        test_cases = generate_verilog_test_cases.invoke(task)
    except Exception:
        test_cases = "Test generation failed:\n" + traceback.format_exc()

    report = f"""
============================================================
VERILOG RTL GENERATION REPORT
============================================================

DESIGN TASK:
{task}

============================================================
GENERATED VERILOG
============================================================

{state.get("code", "No Verilog code generated.")}

============================================================
TEST SCENARIOS
============================================================

{test_cases}

============================================================
END REPORT
============================================================
"""

    print(report)

    return {"report": report}


# ============================================================
# MANAGER NODE
# ============================================================

def manager_decision_node(state: CrewState):
    print("\n" + "=" * 60)
    print("MANAGER DASHBOARD")
    print("=" * 60)

    print(state.get("report", "No report available."))

    command = input(
        "\nCommand (store / another): "
    ).lower().strip()

    if command == "store":
        return {"next_step": "archiver"}

    return {"next_step": "task_input"}


# ============================================================
# ARCHIVER NODE
# ============================================================

def archiver_node(state: CrewState):
    print("\n[ARCHIVER] Verilog task stored successfully.")
    print("Closing workflow.")
    return {"next_step": "exit"}


# ============================================================
# GRAPH
# ============================================================

workflow = StateGraph(CrewState)

workflow.add_node("task_input", task_input_node)
workflow.add_node("developer", real_time_verilog_developer)
workflow.add_node("tester", real_time_verilog_tester)
workflow.add_node("manager_decision", manager_decision_node)
workflow.add_node("archiver", archiver_node)

workflow.add_edge(START, "task_input")


def route_from_input(state: CrewState):
    if state.get("next_step") == "exit":
        return END
    return "developer"


workflow.add_conditional_edges(
    "task_input",
    route_from_input
)

workflow.add_edge("developer", "tester")
workflow.add_edge("tester", "manager_decision")


def route_from_decision(state: CrewState):
    if state.get("next_step") == "archiver":
        return "archiver"
    return "task_input"


workflow.add_conditional_edges(
    "manager_decision",
    route_from_decision
)

workflow.add_edge("archiver", END)

app = workflow.compile()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    print("\nVerilog AI Design Pipeline is ready.")

    try:
        app.invoke(
            {"messages": []},
            config={"recursion_limit": 50},
        )
    except KeyboardInterrupt:
        print("\nStopped by user.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
