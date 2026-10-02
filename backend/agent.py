"""Autonomous Self-Healing Data Analyst Agent.
Writes Python code, executes it in sandbox, retrieves official documentation
via RAG upon encountering errors, and self-heals until clean execution.
"""

import os
import re
from typing import Dict, Any, List, Optional, Callable
import pandas as pd
from backend.code_sandbox import CodeSandbox, ExecutionResult
from backend.rag_engine import rag_engine, RAGEngine


class AnalystAgent:
    """Agent that writes, executes, and self-heals Python code for data analysis."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        provider: str = "auto",  # 'gemini', 'openai', 'heuristic', 'auto'
        model_name: Optional[str] = None
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.provider = provider
        self.model_name = model_name
        self.rag = rag_engine

        # Configure LLM clients if keys are present
        self._init_client()

    def _init_client(self):
        """Initializes API client if keys are available."""
        if not self.api_key:
            # Check environment
            if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
                self.api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
                self.provider = "gemini" if self.provider in ["auto", "gemini"] else self.provider
            elif os.getenv("OPENAI_API_KEY"):
                self.api_key = os.getenv("OPENAI_API_KEY")
                self.provider = "openai" if self.provider in ["auto", "openai"] else self.provider

        if self.provider == "gemini" or (self.provider == "auto" and self.api_key and "AIza" in (self.api_key or "")):
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self.provider = "gemini"
                self.model_name = self.model_name or "gemini-1.5-flash"
            except Exception:
                self.provider = "heuristic"

        elif self.provider == "openai" or (self.provider == "auto" and self.api_key and (self.api_key or "").startswith("sk-")):
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
                self.provider = "openai"
                self.model_name = self.model_name or "gpt-4o-mini"
            except Exception:
                self.provider = "heuristic"
        else:
            if self.provider == "auto":
                self.provider = "heuristic"

    def _call_llm(self, prompt: str) -> str:
        """Invokes LLM or the intelligent heuristic engine."""
        if self.provider == "gemini":
            try:
                import google.generativeai as genai
                model = genai.GenerativeModel(self.model_name or "gemini-1.5-flash")
                response = model.generate_content(prompt)
                return response.text
            except Exception as e:
                # Graceful fallback to heuristic if rate limit or key error
                print(f"Gemini API call failed ({e}), falling back to heuristic engine.")

        elif self.provider == "openai":
            try:
                from openai import OpenAI
                client = getattr(self, "client", OpenAI(api_key=self.api_key))
                resp = client.chat.completions.create(
                    model=self.model_name or "gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2
                )
                return resp.choices[0].message.content or ""
            except Exception as e:
                print(f"OpenAI API call failed ({e}), falling back to heuristic engine.")

        return self._heuristic_generate_code(prompt)

    def _heuristic_generate_code(self, prompt: str) -> str:
        """Intelligent offline code generator that synthesizes Python/Plotly code

        based on dataset schema and user intent without needing external API keys.
        """
        prompt_lower = prompt.lower()

        # Extract column names from prompt if embedded
        cols = []
        cols_match = re.search(r"columns:\s*\[(.*?)\]", prompt, re.IGNORECASE)
        if cols_match:
            cols = [c.strip().strip("'\"") for c in cols_match.group(1).split(",") if c.strip()]

        # Check if this is a healing prompt
        if "error occurred" in prompt_lower or "self-healing" in prompt_lower:
            return self._heuristic_heal_code(prompt)

        # 1. Treemap & Sunburst Hierarchical Visualizations
        if any(w in prompt_lower for w in ["treemap", "sunburst", "hierarchy", "nested"]):
            return """# Hierarchical Treemap Analysis
import plotly.express as px
import pandas as pd
import numpy as np

cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
num_cols = df.select_dtypes(include=np.number).columns.tolist()

if len(cat_cols) >= 2 and num_cols:
    path_cols = cat_cols[:2]
    val_col = num_cols[0]
elif cat_cols and num_cols:
    path_cols = [cat_cols[0]]
    val_col = num_cols[0]
else:
    path_cols = [df.columns[0]]
    val_col = df.columns[-1]

plot_df = df.dropna(subset=path_cols + [val_col]).copy()
plot_df = plot_df[plot_df[val_col] > 0]  # Values must be positive for Treemap

fig = px.treemap(
    plot_df,
    path=path_cols,
    values=val_col,
    color=val_col,
    color_continuous_scale='Blues',
    title=f'Treemap Hierarchy: {" > ".join(path_cols)} by {val_col}',
    template='plotly_white'
)
fig.update_layout(font=dict(family='Inter, sans-serif'))
total_sum = plot_df[val_col].sum()
print(f"Total {val_col} across hierarchy: {total_sum:,.2f}")
"""

        # 2. Donut & Pie Composition Chart
        if any(w in prompt_lower for w in ["donut", "pie", "share", "percentage", "proportion", "composition"]):
            return """# Composition & Share Analysis
import plotly.express as px
import pandas as pd
import numpy as np

cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
num_cols = df.select_dtypes(include=np.number).columns.tolist()

cat_col = cat_cols[0] if cat_cols else df.columns[0]
val_col = num_cols[0] if num_cols else None

if val_col:
    grouped = df.groupby(cat_col, as_index=False)[val_col].sum().sort_values(by=val_col, ascending=False)
else:
    grouped = df[cat_col].value_counts().reset_index()
    grouped.columns = [cat_col, 'Count']
    val_col = 'Count'

fig = px.pie(
    grouped.head(8),
    names=cat_col,
    values=val_col,
    hole=0.45,
    title=f'Proportional Share by {cat_col}',
    template='plotly_white',
    color_discrete_sequence=px.colors.qualitative.Safe
)
fig.update_traces(textposition='inside', textinfo='percent+label')
fig.update_layout(font=dict(family='Inter, sans-serif'))

for _, row in grouped.head(5).iterrows():
    print(f"{row[cat_col]}: {row[val_col]:,.2f}")
"""

        # 3. Violin & Box Multi-Group Distribution
        if any(w in prompt_lower for w in ["violin", "box plot", "boxplot", "variance", "spread across"]):
            return """# Multi-Category Variance & Violin Distribution
import plotly.express as px
import pandas as pd
import numpy as np

cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
num_cols = df.select_dtypes(include=np.number).columns.tolist()

cat_col = cat_cols[0] if cat_cols else df.columns[0]
num_col = num_cols[0] if num_cols else df.columns[-1]

plot_df = df.dropna(subset=[num_col]).copy()
if cat_col in plot_df.columns:
    top_cats = plot_df[cat_col].value_counts().head(8).index
    plot_df = plot_df[plot_df[cat_col].isin(top_cats)]

fig = px.violin(
    plot_df,
    x=cat_col if cat_col in plot_df.columns else None,
    y=num_col,
    color=cat_col if cat_col in plot_df.columns else None,
    box=True,
    points='outliers',
    title=f'Violin Density & IQR Box: {num_col} by {cat_col}',
    template='plotly_white'
)
fig.update_layout(font=dict(family='Inter, sans-serif'))

for cat in plot_df[cat_col].unique()[:5]:
    median_val = plot_df[plot_df[cat_col] == cat][num_col].median()
    print(f"{cat} Median {num_col}: {median_val:,.2f}")
"""

        # 4. Bubble & 4D Multi-Dimensional Chart
        if any(w in prompt_lower for w in ["bubble", "4d", "multi-dimensional", "bubble chart"]):
            return """# 4D Bubble Chart Analysis
import plotly.express as px
import pandas as pd
import numpy as np

num_cols = df.select_dtypes(include=np.number).columns.tolist()
cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

if len(num_cols) >= 3:
    x_col, y_col, size_col = num_cols[0], num_cols[1], num_cols[2]
elif len(num_cols) >= 2:
    x_col, y_col, size_col = num_cols[0], num_cols[1], None
else:
    x_col, y_col, size_col = df.columns[0], df.columns[-1], None

plot_df = df.dropna(subset=[x_col, y_col]).copy()
if size_col:
    plot_df[size_col] = pd.to_numeric(plot_df[size_col], errors='coerce').clip(lower=1).fillna(1)

color_col = cat_cols[0] if (cat_cols and df[cat_cols[0]].nunique() <= 10) else None

fig = px.scatter(
    plot_df,
    x=x_col,
    y=y_col,
    size=size_col,
    color=color_col,
    size_max=35,
    opacity=0.75,
    title=f'Multi-Dimensional Bubble: {x_col} vs {y_col}' + (f' (Size: {size_col})' if size_col else ''),
    template='plotly_white'
)
fig.update_layout(font=dict(family='Inter, sans-serif'))
print(f"Mapped {len(plot_df):,} data points across 4 dimensions.")
"""

        # 5. Pareto 80/20 Analysis Chart
        if any(w in prompt_lower for w in ["pareto", "80/20", "concentration", "cumulative percentage"]):
            return """# Pareto 80/20 Concentration Analysis
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
num_cols = df.select_dtypes(include=np.number).columns.tolist()

cat_col = cat_cols[0] if cat_cols else df.columns[0]
val_col = num_cols[0] if num_cols else df.columns[-1]

grouped = df.groupby(cat_col, as_index=False)[val_col].sum().sort_values(by=val_col, ascending=False).reset_index(drop=True)
grouped['CumPct'] = (grouped[val_col].cumsum() / grouped[val_col].sum()) * 100
top_grouped = grouped.head(15)

fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Bar(x=top_grouped[cat_col], y=top_grouped[val_col], name=val_col, marker_color='#2563EB'), secondary_y=False)
fig.add_trace(go.Scatter(x=top_grouped[cat_col], y=top_grouped['CumPct'], name='Cumulative %', mode='lines+markers', line=dict(color='#EF4444', width=2.5)), secondary_y=True)
fig.add_hline(y=80, line_dash='dash', line_color='#10B981', annotation_text='80% Vital Few', secondary_y=True)

fig.update_layout(title=f'Pareto 80/20 Analysis: {val_col} by {cat_col}', template='plotly_white', font=dict(family='Inter, sans-serif'))
fig.update_yaxes(title_text=f'Total {val_col}', secondary_y=False)
fig.update_yaxes(title_text='Cumulative %', range=[0, 105], secondary_y=True)

vital_few = top_grouped[top_grouped['CumPct'] <= 80]
print(f"Top {len(vital_few)} categories drive ~80% of total {val_col}.")
"""

        # 6. Distribution / Histogram
        if any(w in prompt_lower for w in ["distribution", "histogram", "spread", "skew"]):
            num_cols = [c for c in cols if any(term in c.lower() for term in ["age", "sales", "revenue", "price", "amount", "score", "cost", "total"])]
            target = num_cols[0] if num_cols else (cols[0] if cols else "df.columns[0]")
            return f"""# Distribution Analysis
import plotly.express as px
import pandas as pd
import numpy as np

# Clean and plot
col_name = '{target}' if '{target}' in df.columns else df.select_dtypes(include=np.number).columns[0]
plot_data = df.dropna(subset=[col_name]).copy()

fig = px.histogram(
    plot_data, 
    x=col_name, 
    nbins=30, 
    marginal='box',
    title=f'Distribution and Outliers of {{col_name}}',
    template='plotly_white',
    color_discrete_sequence=['#2563EB']
)
fig.update_layout(xaxis_title=col_name, yaxis_title='Frequency', font=dict(family='Inter, sans-serif'))

mean_val = plot_data[col_name].mean()
median_val = plot_data[col_name].median()
std_val = plot_data[col_name].std()

print(f"Mean {{col_name}}: {{mean_val:,.2f}}")
print(f"Median {{col_name}}: {{median_val:,.2f}}")
print(f"Std Deviation: {{std_val:,.2f}}")
"""

        # 7. Correlation / Scatter / Relationship
        if any(w in prompt_lower for w in ["correlation", "relationship", "scatter", "versus", "vs", "corr"]):
            return """# Correlation & Pattern Analysis
import plotly.express as px
import pandas as pd
import numpy as np

num_cols = df.select_dtypes(include=np.number).columns.tolist()
if len(num_cols) >= 2:
    x_col = num_cols[0]
    y_col = num_cols[1]
    plot_df = df.dropna(subset=[x_col, y_col]).copy()
    
    # Categorical column for color if available
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    color_col = cat_cols[0] if cat_cols and df[cat_cols[0]].nunique() <= 10 else None
    
    fig = px.scatter(
        plot_df, 
        x=x_col, 
        y=y_col, 
        color=color_col,
        title=f'Relationship: {x_col} vs {y_col}',
        template='plotly_white',
        opacity=0.8
    )
    fig.update_layout(font=dict(family='Inter, sans-serif'))
    corr = plot_df[x_col].corr(plot_df[y_col])
    print(f"Pearson Correlation between {x_col} and {y_col}: {corr:.3f}")
else:
    fig = px.bar(df.head(15), title="Data Overview", template="plotly_white")
    print("Insufficient numeric columns for correlation matrix.")
"""

        # 3. Categorical breakdown / Bar chart / GroupBy
        if any(w in prompt_lower for w in ["category", "breakdown", "by region", "by department", "churn", "segment", "bar", "top"]):
            return """# Grouped Aggregation & Category Comparison
import plotly.express as px
import pandas as pd
import numpy as np

cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
num_cols = df.select_dtypes(include=np.number).columns.tolist()

group_col = cat_cols[0] if cat_cols else df.columns[0]
metric_col = num_cols[0] if num_cols else df.columns[-1]

# Aggregate with reset_index to avoid KeyError in Plotly
grouped = df.groupby(group_col, as_index=False)[metric_col].agg(['mean', 'count']).reset_index()
grouped = grouped.sort_values(by='mean', ascending=False).head(10)

fig = px.bar(
    grouped,
    x=group_col,
    y='mean',
    text='mean',
    title=f'Average {metric_col} by {group_col} (Top 10)',
    template='plotly_white',
    color='mean',
    color_continuous_scale='Blues'
)
fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
fig.update_layout(xaxis_title=group_col, yaxis_title=f'Mean {metric_col}', font=dict(family='Inter, sans-serif'))

for _, row in grouped.iterrows():
    print(f"{row[group_col]}: Mean = {row['mean']:,.2f} (Count: {int(row['count'])})")
"""

        # 4. Default Comprehensive Analysis
        return """# Comprehensive Automated Analysis
import plotly.express as px
import pandas as pd
import numpy as np

num_cols = df.select_dtypes(include=np.number).columns.tolist()
cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

if num_cols and cat_cols:
    cat_col = cat_cols[0]
    num_col = num_cols[0]
    agg_df = df.groupby(cat_col, as_index=False)[num_col].sum().sort_values(by=num_col, ascending=False).head(12)
    fig = px.bar(
        agg_df, 
        x=cat_col, 
        y=num_col, 
        title=f'Total {num_col} by {cat_col}',
        template='plotly_white',
        color_discrete_sequence=['#4F46E5']
    )
elif len(num_cols) >= 2:
    fig = px.scatter(
        df, 
        x=num_cols[0], 
        y=num_cols[1], 
        title=f'{num_cols[0]} vs {num_cols[1]}',
        template='plotly_white'
    )
else:
    col = df.columns[0]
    counts = df[col].value_counts().head(10).reset_index()
    counts.columns = [col, 'Count']
    fig = px.bar(counts, x=col, y='Count', title=f'Top {col} Frequencies', template='plotly_white')

fig.update_layout(font=dict(family='Inter, sans-serif'))
print("Analysis executed successfully.")
"""

    def _heuristic_heal_code(self, prompt: str) -> str:
        """Applies rule-based AST/regex code transformations for common errors

        when running in offline mode.
        """
        # Extract code from prompt
        code_match = re.search(r"```python(.*?)```", prompt, re.DOTALL)
        if not code_match:
            code_match = re.search(r"```(.*?)```", prompt, re.DOTALL)
        prev_code = code_match.group(1).strip() if code_match else prompt

        # Strip whitespace from columns at start
        fix_prefix = "df.columns = df.columns.str.strip()\n"
        healed_code = prev_code

        # Fix 1: Chained indexing or SettingWithCopy
        if "SettingWithCopy" in prompt or "copy" in prompt:
            healed_code = re.sub(r"(\w+)\[(.*?)\]\[(.*?)\]\s*=", r"\1.loc[\2, \3] =", healed_code)

        # Fix 2: groupby without reset_index
        if "KeyError" in prompt and "groupby" in healed_code:
            healed_code = re.sub(r"\.groupby\((.*?)\)\[(.*?)\]\.(mean|sum|count)\(\)", r".groupby(\1, as_index=False)[\2].\3()", healed_code)

        # Fix 3: replace append with pd.concat
        if "append" in prompt:
            healed_code = re.sub(r"(\w+)\.append\((.*?)\)", r"pd.concat([\1, \2], ignore_index=True)", healed_code)

        # Fix 4: datetime .dt on string
        if "AttributeError" in prompt and ".dt." in healed_code:
            # Add pd.to_datetime conversion
            date_col_match = re.search(r"df\[['\"](.*?)['\"]\]\.dt\.", healed_code)
            if date_col_match:
                dcol = date_col_match.group(1)
                fix_prefix += f"df['{dcol}'] = pd.to_datetime(df['{dcol}'], errors='coerce')\n"

        if not healed_code.startswith("df.columns"):
            healed_code = fix_prefix + healed_code

        return healed_code

    def _extract_code_block(self, text: str) -> str:
        """Extracts python code from markdown fences or raw text."""
        code_match = re.search(r"```python\s*(.*?)\s*```", text, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
        code_match = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
        return text.strip()

    def run_analysis(
        self,
        df: pd.DataFrame,
        query: str,
        max_retries: int = 4,
        on_step_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> Dict[str, Any]:
        """Executes an autonomous analysis cycle with RAG error recovery and self-healing.

        Args:
            df: Target pandas DataFrame.
            query: User's analytical goal or question.
            max_retries: Maximum number of self-healing iterations.
            on_step_callback: Real-time progress updates callback.

        Returns:
            Dictionary with final execution results, figures, healing logs, and insights.
        """
        def emit_step(stage: str, message: str, details: Optional[Dict[str, Any]] = None):
            if on_step_callback:
                on_step_callback({
                    "stage": stage,
                    "message": message,
                    "details": details or {}
                })

        # Step 1: Profile Schema Context
        col_summary = []
        for col in df.columns:
            col_summary.append(f"- '{col}' ({df[col].dtype}): {df[col].nunique()} unique, {df[col].isnull().sum()} nulls")
        schema_context = "\n".join(col_summary)
        head_sample = df.head(3).to_dict(orient="records")

        emit_step("planning", f"Formulating Python analysis for: '{query}'")

        # Step 2: Initial Prompt Formulation
        base_prompt = f"""You are an autonomous Principal Data Scientist and Python/Pandas expert.
Write clean, executable Python code to solve the user's analytical question:
"{query}"

Dataset Schema:
{schema_context}

Sample Data (first 3 rows):
{head_sample}

Requirements:
1. The DataFrame is pre-loaded as variable `df`. Do NOT read from disk or define a dummy df.
2. Use pandas, numpy, and plotly.express (as px) or plotly.graph_objects (as go).
3. Always store the primary interactive chart in variable `fig`.
4. Always call `.reset_index()` after groupby or aggregations so column names exist for Plotly!
5. Print 2-4 key numerical insights or statistical takeaways using `print(...)`.
6. Return ONLY valid Python code inside ```python ``` code block. No extraneous conversational filler.
"""

        healing_history: List[Dict[str, Any]] = []
        current_code = self._extract_code_block(self._call_llm(base_prompt))
        attempt = 0
        execution_result: Optional[ExecutionResult] = None

        while attempt <= max_retries:
            attempt += 1
            emit_step("executing", f"Executing code (Attempt {attempt}/{max_retries + 1})...", {"code": current_code})

            execution_result = CodeSandbox.execute(current_code, df)

            if execution_result.success:
                emit_step("success", f"Execution succeeded on Attempt {attempt}!", {
                    "figures_count": len(execution_result.figures),
                    "stdout": execution_result.stdout
                })
                break

            # Execution Failed: Begin RAG-driven Self-Healing
            emit_step("error_detected", f"Encountered {execution_result.error_type}: {execution_result.error_message}", {
                "error_type": execution_result.error_type,
                "error_message": execution_result.error_message,
                "failing_line": execution_result.failing_line,
                "failing_code": execution_result.failing_code
            })

            if attempt > max_retries:
                emit_step("max_retries", "Reached maximum self-healing attempts.")
                break

            # RAG Retrieval from official documentation
            search_query = f"{execution_result.error_message} {execution_result.failing_code or ''}"
            retrieved_docs = self.rag.retrieve(search_query, error_type=execution_result.error_type or "", top_k=2)
            rag_context_text = self.rag.format_rag_context(retrieved_docs)

            emit_step("rag_retrieval", f"Retrieved {len(retrieved_docs)} official doc topics for error recovery", {
                "docs": [{"title": d["title"], "category": d["category"]} for d in retrieved_docs]
            })

            # Record in healing history
            healing_record = {
                "attempt": attempt,
                "error_type": execution_result.error_type,
                "error_message": execution_result.error_message,
                "failing_line": execution_result.failing_line,
                "failing_code": execution_result.failing_code,
                "retrieved_topics": [d["title"] for d in retrieved_docs],
                "previous_code": current_code
            }

            # Self-healing synthesis prompt
            healing_prompt = f"""You are debugging a Python data analysis script that threw an error during execution.
Fix the bug and provide the complete, corrected Python code.

User's Original Analytical Goal:
"{query}"

Dataset Schema:
{schema_context}

Previous Code That Threw Error:
```python
{current_code}
```

Error Details:
Error Type: {execution_result.error_type}
Error Message: {execution_result.error_message}
Failing Line Number: {execution_result.failing_line}
Failing Code Snippet: {execution_result.failing_code}

Full Traceback:
{execution_result.traceback_str}

{rag_context_text}

Correction Instructions:
1. Review the official documentation above and fix the root cause.
2. Ensure column names match actual dataset schema exactly.
3. Make sure `fig` contains the Plotly figure and call `.reset_index()` if using groupby.
4. Output ONLY the complete corrected Python code inside ```python ``` blocks.
"""
            emit_step("healing", f"Self-healing code with RAG guidance (Iteration {attempt})...")
            new_code_raw = self._call_llm(healing_prompt)
            current_code = self._extract_code_block(new_code_raw)
            healing_record["healed_code"] = current_code
            healing_history.append(healing_record)

        # Synthesize final narrative insights from stdout and results
        insights_narrative = self._generate_insights_narrative(query, execution_result)

        return {
            "success": execution_result.success if execution_result else False,
            "final_code": current_code,
            "figures": execution_result.figures if execution_result else [],
            "tables": execution_result.tables if execution_result else {},
            "stdout": execution_result.stdout if execution_result else "",
            "error_type": execution_result.error_type if execution_result and not execution_result.success else None,
            "error_message": execution_result.error_message if execution_result and not execution_result.success else None,
            "healing_history": healing_history,
            "total_attempts": attempt,
            "insights": insights_narrative
        }

    def _generate_insights_narrative(self, query: str, res: Optional[ExecutionResult]) -> List[str]:
        """Converts printed outputs and numerical results into professional, clear bullet points."""
        insights = []
        if not res or not res.success:
            return ["Analysis could not complete successfully after self-healing attempts."]

        if res.stdout.strip():
            lines = [line.strip() for line in res.stdout.strip().split("\n") if line.strip()]
            for line in lines:
                if not line.lower().startswith("traceback") and not line.lower().startswith("warning"):
                    insights.append(line)

        if not insights:
            insights.append(f"Successfully processed dataset and generated visual insights for '{query}'.")
            if res.figures:
                insights.append(f"Generated {len(res.figures)} interactive visualization(s) showing distribution and patterns.")

        return insights
