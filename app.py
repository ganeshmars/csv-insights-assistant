import streamlit as st
import pandas as pd
import plotly.express as px
import os
from dotenv import load_dotenv
import tempfile
import re
import json  # For parsing JSON from classifier
from typing import TypedDict, Union, List, Dict, Any  # Added Dict, Any for more precise typing
import operator  # Not explicitly used in this version, but often useful with LangGraph state

# Langchain & LangGraph components
from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain.agents.agent_types import AgentType
from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage  # HumanMessage not directly used here but good to know
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chains import LLMChain

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY not found. Please set it in your .env file or environment.")
    st.stop()

LLM_MODEL = "gpt-3.5-turbo-0125"  # Or gpt-4 for better classification/generation

# --- LLM Instances ---
classifier_llm = ChatOpenAI(model=LLM_MODEL, temperature=0, openai_api_key=OPENAI_API_KEY)
generator_llm = ChatOpenAI(model=LLM_MODEL, temperature=0,
                           openai_api_key=OPENAI_API_KEY)  # Used by LLMChain and create_csv_agent


# --- Helper Functions ---
def safe_exec_plotly(code_string: str, df: pd.DataFrame):
    local_vars = {"pd": pd, "px": px, "df": df, "fig": None}
    global_vars = {}
    try:
        exec(code_string, global_vars, local_vars)
        return local_vars.get("fig")
    except Exception as e:
        st.error(f"Error executing Plotly code: {e}")
        st.code(code_string)
        return None


def extract_code_from_markdown(md_string: str) -> str:
    if not md_string: return ""
    match = re.search(r"```python\n(.*?)\n```", md_string, re.DOTALL)
    if match:
        return match.group(1).strip()
    match_plain = re.search(r"```\n(.*?)\n```", md_string, re.DOTALL)
    if match_plain:
        return match_plain.group(1).strip()
    if "import plotly.express as px" in md_string and "fig =" in md_string:
        lines = md_string.split('\n')
        code_lines = [line for line in lines if not line.strip().startswith("#")]
        potential_code = "\n".join(code_lines).strip()
        if all(kw in potential_code for kw in ["import plotly.express as px", "fig ="]):
            non_code_words = re.sub(
                r'\b(import|from|as|def|class|if|else|elif|for|while|try|except|finally|return|px|pd|df|fig)\b', '',
                potential_code, flags=re.IGNORECASE)
            non_code_words = re.sub(r'[=\(\)\{\}\[\],:\.\'"]', '', non_code_words)
            if len(non_code_words.split()) < 20:  # Increased threshold slightly
                return potential_code
    return md_string.strip()


def get_dataframe_context_for_llm(csv_path: str, num_rows: int = 3, include_columns: bool = True) -> str:
    """Gets a string representation of DataFrame head and/or columns for LLM context."""
    try:
        df_sample = pd.read_csv(csv_path, nrows=num_rows)
        context_parts = []
        if include_columns:
            context_parts.append(f"DataFrame Columns: {df_sample.columns.tolist()}")
        if num_rows > 0:
            context_parts.append(f"DataFrame Head (first {num_rows} rows):\n{df_sample.to_string()}")
        if not context_parts:
            return "No dataframe context available (file might be empty or unreadable for preview)."
        return "\n\n".join(context_parts)
    except Exception as e:
        return f"Could not load DataFrame preview: {e}"


# --- LangGraph State Definition ---
class AgentState(TypedDict):
    original_query: str
    csv_file_path: str
    df_for_plotting: Union[pd.DataFrame, None]
    classification_result: Dict[
        str, Any]  # Expecting keys like "type", "text_task", "plot_task", "text_task_type_for_both"
    text_insight: Union[str, None]
    plotly_code: Union[str, None]
    error_message: Union[str, None]
    final_response_parts: List[Dict[str, Any]]


# --- LangGraph Nodes ---
def classify_query_node(state: AgentState):
    st.write("Step 1: Classifying query...")
    query = state["original_query"]
    try:
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=f"""You are a query classification expert for a CSV data analysis assistant.
Your task is to analyze the user's query and determine the type of response and specific sub-tasks.
The dataframe `df` will be available for later processing by other components.

Possible output types:
1. 'TEXT': For simple textual answers, definitions, or information directly inferable from a small data preview (e.g., column names, value in a previewed row, existence of a column).
2. 'COMPLEX_TEXT_CALCULATION': For textual answers that require calculations or aggregations over the ENTIRE dataset (e.g., total sum, average, count of unique values across all rows).
3. 'PLOT': For generating a Plotly graph visualization.
4. 'BOTH': If the query requires both a plot AND a textual answer.

Output a JSON object with the following structure:
{{
  "type": "TEXT" | "COMPLEX_TEXT_CALCULATION" | "PLOT" | "BOTH",
  "text_task_type_for_both": "TEXT" | "COMPLEX_TEXT_CALCULATION" | null, // Relevant only if "type" is "BOTH" and a text_task exists. Null otherwise.
  "text_task": "Specific instruction for the textual part (simple or complex), or null if 'PLOT' only.",
  "plot_task": "Specific instruction for generating Plotly code, or null if text-only."
}}

Examples:
Query: "What are the column names?"
Output: {{"type": "TEXT", "text_task_type_for_both": null, "text_task": "List the column names from the dataset.", "plot_task": null}}

Query: "What is the total sales?"
Output: {{"type": "COMPLEX_TEXT_CALCULATION", "text_task_type_for_both": null, "text_task": "Calculate the total sales by summing the relevant sales/amount column from the entire dataset and provide the result.", "plot_task": null}}

Query: "Plot a histogram of ages."
Output: {{"type": "PLOT", "text_task_type_for_both": null, "text_task": null, "plot_task": "Generate Python code for a Plotly Express histogram of the 'age' column."}}

Query: "Show me the average price per category and also plot it as a bar chart."
Output: {{"type": "BOTH", "text_task_type_for_both": "COMPLEX_TEXT_CALCULATION", "text_task": "Calculate the average price for each category from the entire dataset and list them.", "plot_task": "Generate Python code for a Plotly Express bar chart showing average price by category."}}

Query: "What does SKU mean and plot sales by SKU?"
Output: {{"type": "BOTH", "text_task_type_for_both": "TEXT", "text_task": "Provide a definition or explanation of 'SKU'.", "plot_task": "Generate a plot of sales by SKU."}}

User Query: "{query}"
"""),
        ])
        chain = prompt | classifier_llm
        response = chain.invoke({})
        response_content_str = response.content
        if not isinstance(response_content_str, str):
            response_content_str = str(response_content_str)
        if response_content_str.strip().startswith("```json"):
            response_content_str = response_content_str.strip()[7:-3]
        elif response_content_str.strip().startswith("```"):
            response_content_str = response_content_str.strip()[3:-3]
        response_content_str = response_content_str.strip()
        st.write(f"Raw JSON string from LLM: ```\n{response_content_str}\n```")
        classification = json.loads(response_content_str)
        st.write(f"Classification: {classification}")
        return {"classification_result": classification, "error_message": None}
    except json.JSONDecodeError as je:
        st.error(f"Error decoding JSON from classifier: {je}")
        st.text(
            f"Problematic JSON string was: {response_content_str if 'response_content_str' in locals() else 'Not available'}")
        return {"error_message": f"Classification JSON decoding failed: {je}",
                "classification_result": {"type": "ERROR"}}
    except Exception as e:
        st.error(f"Error in classification: {e}")
        return {"error_message": f"Classification failed: {e}", "classification_result": {"type": "ERROR"}}


def generate_text_insight_node(state: AgentState):  # Simple text using LLMChain
    if state.get("error_message"): return {}
    task = state["classification_result"].get("text_task")
    if not task: return {"text_insight": None}
    st.write("Step 2a: Generating simple textual insight (using LLMChain)...")
    csv_path = state["csv_file_path"]
    try:
        df_context_for_llm = get_dataframe_context_for_llm(csv_path, num_rows=3, include_columns=True)
        text_prompt_str = """
You are a data analyst. You are given a task and some context from a CSV file's DataFrame.
DataFrame Context:
---
{dataframe_context}
---
Task: {llm_task}
Based on the task and the provided DataFrame context, provide a concise, direct textual answer.
If the task involves a calculation that can be inferred from the context or is simple (e.g., counting visible items, identifying a column name), perform it.
If the task requires complex calculations beyond simple inference from the preview (e.g., summing a whole column, complex aggregations), state that you would need to perform a more detailed analysis on the full dataset.
Focus on answering the specific task. Answer:"""
        text_prompt = PromptTemplate(input_variables=["llm_task", "dataframe_context"], template=text_prompt_str)
        llm_chain = LLMChain(llm=generator_llm, prompt=text_prompt,
                             verbose=st.session_state.get("verbose_logging", False))
        response = llm_chain.invoke({"llm_task": task, "dataframe_context": df_context_for_llm})
        insight = response.get("text",
                               str(response) if isinstance(response, dict) else response)  # More robust extraction
        insight = insight.strip() if insight else "Could not generate simple text insight."
        st.write(f"Simple Text Insight (from LLMChain): {insight}")
        return {"text_insight": insight, "error_message": None}
    except Exception as e:
        st.error(f"Error generating simple text insight with LLMChain: {e}")
        return {"text_insight": f"Error with LLMChain: {e}",
                "error_message": f"Simple text insight (LLMChain) failed: {e}"}


def generate_computational_text_node(state: AgentState):
    if state.get("error_message"): return {}
    task = state["classification_result"].get("text_task")
    if not task: return {"text_insight": None}
    st.write("Step 2b: Generating computational text (using create_csv_agent)...")
    csv_path = state["csv_file_path"]
    try:
        agent_prompt = f"""
You are a data analyst agent with access to a pandas DataFrame ('df') from a CSV. Your task is: {task}
You MUST use your tools (like the python_repl_ast) to perform any necessary calculations on the DataFrame.
After your calculations and internal thought process, your final response to me (the user of your output) MUST be ONLY in the format:
Final Answer: [Your concise, direct textual answer, which is the result of the calculation]
Do NOT include your "Thought:", "Action:", "Action Input:", or "Observation:" steps in this final output to me. Just the "Final Answer: [result]".
Current Task: "{task}". What is the result?"""
        csv_agent = create_csv_agent(
            generator_llm, csv_path, verbose=st.session_state.get("verbose_logging", False),
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION, allow_dangerous_code=True, handle_parsing_errors=True
        )
        response = csv_agent.invoke({"input": agent_prompt})
        insight = response.get("output", "Could not generate computational text insight.")
        if isinstance(insight, str) and insight.strip().upper().startswith("FINAL ANSWER:"):
            insight = insight.strip()[len("FINAL ANSWER:"):].strip()
        st.write(f"Computational Text Insight: {insight}")
        return {"text_insight": insight, "error_message": None}
    except Exception as e:
        st.error(f"Error generating computational text insight: {e}")
        return {"text_insight": f"Error in computational text: {e}",
                "error_message": f"Computational text generation failed: {e}"}


def generate_plotly_code_node(state: AgentState):
    if state.get("error_message"): return {}
    task = state["classification_result"].get("plot_task")
    if not task: return {"plotly_code": None}
    st.write("Step 2c: Generating Plotly code (using create_csv_agent)...")
    csv_path = state["csv_file_path"]
    try:
        agent_prompt = f"""
You are a Python data visualization expert. Given the dataframe `df` from the CSV: Task: {task}
Generate ONLY the Python code to create a Plotly Express figure. The code MUST use `plotly.express` (imported as `px`).
The pandas DataFrame is available as `df`. The final Plotly figure object MUST be assigned to a variable named `fig`.
Do NOT include any explanations, comments outside the code, or surrounding text. Example:
```python
import plotly.express as px
fig = px.bar(df, x='category_column', y='value_column', title='My Plot')
```"""
        csv_agent = create_csv_agent(
            generator_llm, csv_path, verbose=st.session_state.get("verbose_logging", False),
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION, allow_dangerous_code=True, handle_parsing_errors=True
        )
        response = csv_agent.invoke({"input": agent_prompt})
        raw_code_output = response.get("output", "")
        plotly_code = extract_code_from_markdown(raw_code_output)
        if not plotly_code or "fig =" not in plotly_code:
            st.warning(f"Plotly code might be missing/malformed. Raw: {raw_code_output}")
            commented_raw_code = "\n".join([f"# {line}" for line in raw_code_output.split('\n')])
            if not plotly_code: plotly_code = f"# Agent failed to produce valid Plotly code.\n# Raw output:\n{commented_raw_code}"
        st.write(f"Plotly Code Extracted: \n{plotly_code}")
        return {"plotly_code": plotly_code, "error_message": None}
    except Exception as e:
        st.error(f"Error generating Plotly code: {e}")
        commented_exception = "\n".join([f"# {line}" for line in str(e).split('\n')])
        return {"plotly_code": f"# Error during Plotly code generation:\n{commented_exception}",
                "error_message": f"Plotly code gen failed: {e}"}


def compile_response_node(state: AgentState):
    st.write("Step 3: Compiling response...")
    parts = []
    if state.get("error_message") and not (state.get("text_insight") or state.get("plotly_code")):
        parts.append({"type": "error", "content": state["error_message"]})
    else:
        if state.get("error_message"):
            parts.append({"type": "error",
                          "content": f"Warning: An error occurred: {state['error_message']}. Displaying partial results."})
        if state.get("text_insight"):
            parts.append({"type": "text", "content": state["text_insight"]})
        if state.get("plotly_code"):
            df_plot = state.get("df_for_plotting")
            if df_plot is None and state.get("csv_file_path"):
                try:
                    df_plot = pd.read_csv(state["csv_file_path"])
                    # Update state so it's available if this node is re-run or for other uses
                    state["df_for_plotting"] = df_plot
                except Exception as e:
                    parts.append({"type": "error", "content": f"Failed to load DataFrame for plotting: {e}"})
                    df_plot = None  # Ensure df_plot is None if loading failed

            if df_plot is not None:
                parts.append({"type": "plotly_code", "content": state["plotly_code"], "df": df_plot})
            else:
                parts.append({"type": "text", "content": "Could not load data to render the plot."})  # Or error
    st.write(
        f"Final parts for UI: {[(p['type'], p['content'][:50] + '...' if p['type'] == 'text' or p['type'] == 'plotly_code' else p['content']) for p in parts]}")
    return {"final_response_parts": parts}


# --- LangGraph Workflow Definition ---
workflow = StateGraph(AgentState)
workflow.add_node("classifier", classify_query_node)
workflow.add_node("simple_text_generator", generate_text_insight_node)  # Renamed for clarity
workflow.add_node("computational_text_generator", generate_computational_text_node)
workflow.add_node("plot_generator", generate_plotly_code_node)
workflow.add_node("compiler", compile_response_node)

workflow.set_entry_point("classifier")


def decide_next_step(state: AgentState):
    if state.get("error_message") or state["classification_result"].get("type") == "ERROR":
        return "compiler"

    classification = state["classification_result"]
    classification_type = classification["type"]

    if classification_type == "TEXT":
        return "simple_text_generator"
    elif classification_type == "COMPLEX_TEXT_CALCULATION":
        return "computational_text_generator"
    elif classification_type == "PLOT":
        return "plot_generator"
    elif classification_type == "BOTH":
        # For "BOTH", decide the first step based on text_task_type_for_both
        # The subsequent step (plot or compile) will be handled by the node's conditional edge
        text_task_type = classification.get("text_task_type_for_both")
        if classification.get("text_task"):  # If there's a text task involved in BOTH
            if text_task_type == "COMPLEX_TEXT_CALCULATION":
                return "computational_text_generator"
            elif text_task_type == "TEXT":  # Simple text
                return "simple_text_generator"
            else:  # Fallback if text_task_type_for_both is not set but text_task exists
                st.warning("`text_task_type_for_both` missing for 'BOTH' type, using heuristic for text_task.")
                text_task_lower = classification["text_task"].lower()
                if any(keyword in text_task_lower for keyword in
                       ["calculate", "total", "average", "sum", "count", "how many", "unique values"]):
                    return "computational_text_generator"
                else:
                    return "simple_text_generator"
        elif classification.get("plot_task"):  # If "BOTH" but only plot_task is defined (should be rare)
            return "plot_generator"
        else:  # "BOTH" with no tasks, go to compiler
            return "compiler"
    return "compiler"  # Fallback


workflow.add_conditional_edges("classifier", decide_next_step, {
    "simple_text_generator": "simple_text_generator",
    "computational_text_generator": "computational_text_generator",
    "plot_generator": "plot_generator",
    "compiler": "compiler"
})


def after_text_node_logic(state: AgentState):  # Common logic for after any text node
    if state.get("error_message"): return "compiler"
    # If it was 'BOTH' and a plot task exists, go to plot generator, else compile
    if state["classification_result"]["type"] == "BOTH" and state["classification_result"].get("plot_task"):
        return "plot_generator"
    return "compiler"


workflow.add_conditional_edges("simple_text_generator", after_text_node_logic, {
    "plot_generator": "plot_generator", "compiler": "compiler"
})
workflow.add_conditional_edges("computational_text_generator", after_text_node_logic, {
    "plot_generator": "plot_generator", "compiler": "compiler"
})

workflow.add_edge("plot_generator", "compiler")
workflow.add_edge("compiler", END)

app_graph = workflow.compile()

# --- Streamlit App UI ---
st.set_page_config(page_title="CSV Insights & Graph Assistant (LangGraph)", layout="wide")
st.title("📊 CSV Insights & Graph Assistant (LangGraph Version)")
st.markdown("Upload a CSV file and ask questions. The assistant will determine if you need text, a graph, or both!")

if "verbose_logging" not in st.session_state: st.session_state.verbose_logging = False
if "csv_file_path" not in st.session_state: st.session_state.csv_file_path = None

with st.sidebar:
    st.header("Configuration")
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    st.session_state.verbose_logging = st.checkbox("Enable Verbose Agent Logging",
                                                   value=st.session_state.verbose_logging)

    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            st.session_state.csv_file_path = tmp_file.name
        st.success(f"Uploaded '{uploaded_file.name}'")
        try:
            df_preview = pd.read_csv(st.session_state.csv_file_path, nrows=5)
            st.subheader("Data Preview (First 5 rows):")
            st.dataframe(df_preview)
        except Exception as e:
            st.error(f"Error reading CSV for preview: {e}")
            if st.session_state.csv_file_path and os.path.exists(st.session_state.csv_file_path):
                os.remove(st.session_state.csv_file_path)  # Clean up bad temp file
            st.session_state.csv_file_path = None
    # If no new file is uploaded, st.session_state.csv_file_path retains its value from previous run (if any)

if st.session_state.csv_file_path:
    user_query = st.text_input("Ask something about your data:", key="query_input_lg")
    if st.button("Get Insights (LangGraph)", key="submit_button_lg"):
        if not user_query:
            st.warning("Please enter a question.")
        else:
            with st.spinner("AI is thinking with LangGraph... This may take a moment."):
                try:
                    initial_state = AgentState(
                        original_query=user_query, csv_file_path=st.session_state.csv_file_path,
                        df_for_plotting=None, classification_result={}, text_insight=None,
                        plotly_code=None, error_message=None, final_response_parts=[]
                    )
                    st.markdown("---");
                    st.subheader("AI Processing Log:")
                    final_state = app_graph.invoke(initial_state)
                    st.markdown("---");
                    st.subheader("Assistant's Final Response:")

                    if final_state.get("final_response_parts"):
                        for part in final_state["final_response_parts"]:
                            if part["type"] == "text":
                                st.markdown(part["content"])
                            elif part["type"] == "plotly_code":
                                st.info("Attempting to render Plotly graph...")
                                st.code(part["content"], language="python")
                                df_plot = part.get("df")  # df should be passed by compiler
                                if df_plot is not None:
                                    plotly_fig = safe_exec_plotly(part["content"], df_plot)
                                    if plotly_fig:
                                        st.plotly_chart(plotly_fig, use_container_width=True)
                                    else:
                                        st.warning("Could not render the Plotly graph from the generated code.")
                                else:
                                    st.warning("DataFrame not available for plotting this part.")
                            elif part["type"] == "error":
                                st.error(part["content"])
                    else:
                        st.warning("No response parts were generated.")
                except Exception as e:
                    st.error(f"A critical error occurred in the LangGraph execution: {e}")
                    import traceback

                    st.code(traceback.format_exc())
else:
    st.info("Upload a CSV file and enter a query to get started.")