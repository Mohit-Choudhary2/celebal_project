"""Official documentation knowledge base for Python, Pandas, NumPy, and Plotly.
Provides structured doc chunks, syntax patterns, error signatures, and recovery solutions
for the Autonomous Self-Healing Data Analyst Agent.
"""

from typing import List, Dict, Any

OFFICIAL_DOC_CHUNKS: List[Dict[str, Any]] = [
    {
        "id": "pandas_keyerror_columns",
        "category": "Pandas Indexing & Column Access",
        "error_types": ["KeyError"],
        "keywords": ["keyerror", "column not found", "column", "case sensitivity", "index", "columns"],
        "title": "Pandas Column Access & KeyError Resolution",
        "doc_content": """
OFFICIAL DOCS: pandas.DataFrame[] / KeyError
A KeyError occurs when attempting to access a column name or index label that does not exist in DataFrame.columns.

Common causes:
1. Whitespace or trailing spaces in column names: e.g., ' Sales ' instead of 'Sales'.
2. Case sensitivity: 'revenue' vs 'Revenue' or 'REVENUE'.
3. Operating on columns that were dropped or renamed earlier.
4. Using .loc with boolean mask or missing label.

Recommended fixes:
- Strip whitespace from column names: `df.columns = df.columns.str.strip()`
- Lowercase column names for uniform access: `df.columns = df.columns.str.strip().str.lower()`
- Check available columns: `list(df.columns)` or `df.columns.tolist()`
- Safely check before access: `if 'col_name' in df.columns:`
- Use `df.get('col_name', default_value)` for graceful fallback.
""",
        "example_bad": "df['sales ']  # KeyError if actual column is 'sales'",
        "example_good": "df.columns = df.columns.str.strip()\nsales = df['sales']"
    },
    {
        "id": "pandas_settingwithcopy",
        "category": "Pandas Indexing & Assignment",
        "error_types": ["SettingWithCopyWarning", "SettingWithCopy"],
        "keywords": ["settingwithcopywarning", "chained assignment", "copy", "slice", "loc"],
        "title": "Pandas SettingWithCopyWarning & Chained Assignment",
        "doc_content": """
OFFICIAL DOCS: pandas.DataFrame.loc / SettingWithCopyWarning
A value is trying to be set on a copy of a slice from a DataFrame.
Occurs when using chained indexing like `df[df['A'] > 2]['B'] = 3`. Pandas cannot determine whether
the slice is a view or a copy, leading to unpredictable assignment.

Recommended fixes:
- Always use `.loc[row_indexer, col_indexer]` for assignment:
  `df.loc[df['A'] > 2, 'B'] = 3`
- If you explicitly want an independent copy after filtering:
  `sub_df = df[df['A'] > 2].copy()`
  `sub_df['B'] = 3`
""",
        "example_bad": "filtered = df[df['status'] == 'Active']\nfiltered['score'] = 100",
        "example_good": "filtered = df[df['status'] == 'Active'].copy()\nfiltered.loc[:, 'score'] = 100"
    },
    {
        "id": "pandas_groupby_agg",
        "category": "Pandas GroupBy & Aggregations",
        "error_types": ["ValueError", "TypeError", "AttributeError", "KeyError"],
        "keywords": ["groupby", "agg", "aggregation", "multiindex", "as_index", "observed"],
        "title": "Pandas GroupBy Aggregations & MultiIndex Flattening",
        "doc_content": """
OFFICIAL DOCS: pandas.DataFrame.groupby / pandas.DataFrame.agg
When aggregating after groupby, non-numeric columns can cause `TypeError: agg function failed` if applying
numeric aggregations like 'mean', 'sum', or 'std'. Also, aggregating multiple columns generates a MultiIndex.

Recommended fixes:
1. Specify `numeric_only=True` or explicitly select numeric columns before calculating mean/sum:
   `df.groupby('category')[['sales', 'profit']].mean().reset_index()`
   `df.groupby('category', as_index=False).agg({'sales': 'sum', 'profit': 'mean'})`
2. Always call `.reset_index()` so the grouping column remains accessible for plotting with Plotly/Seaborn.
3. If MultiIndex columns are created by `.agg(['mean', 'std'])`, flatten them:
   `grouped.columns = ['_'.join(c).strip('_') for c in grouped.columns]`
4. In categorical groupings, specify `observed=False` or `observed=True` to avoid FutureWarning.
""",
        "example_bad": "df.groupby('region').mean()  # Fails if string columns exist without numeric_only",
        "example_good": "df.groupby('region', as_index=False)[['sales', 'revenue']].mean()"
    },
    {
        "id": "pandas_datetime_conversion",
        "category": "Pandas Datetime Operations",
        "error_types": ["ValueError", "TypeError", "AttributeError"],
        "keywords": ["datetime", "date", "to_datetime", "dt", "strftime", "resample", "resampling"],
        "title": "Pandas Datetime Parsing & dt Accessor",
        "doc_content": """
OFFICIAL DOCS: pandas.to_datetime / Series.dt accessor
Dates stored as strings or mixed formats often cause errors when extracting month, year, or resampling.
Calling `.dt` on non-datetime Series raises `AttributeError: Can only use .dt accessor with datetimelike values`.

Recommended fixes:
1. Convert column using `pd.to_datetime`:
   `df['date'] = pd.to_datetime(df['date'], errors='coerce')`
   `errors='coerce'` converts unparseable dates to NaT instead of raising ValueError.
2. Filter or handle NaT before date-specific operations:
   `valid_dates = df.dropna(subset=['date'])`
3. Extract date components using `.dt`:
   `df['year'] = df['date'].dt.year`
   `df['month'] = df['date'].dt.month`
   `df['year_month'] = df['date'].dt.to_period('M').astype(str)`
4. For time-series resampling, ensure datetime is the index:
   `df.set_index('date').resample('ME')['sales'].sum().reset_index()`
   (Note: in Pandas >= 2.2, 'M' resample is deprecated in favor of 'ME' for month-end).
""",
        "example_bad": "df['month'] = df['order_date'].dt.month  # Fails if order_date is object/str",
        "example_good": "df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')\ndf['month'] = df['order_date'].dt.month"
    },
    {
        "id": "pandas_concat_append",
        "category": "Pandas DataFrame Reshaping & Combining",
        "error_types": ["AttributeError"],
        "keywords": ["append", "concat", "combine", "dataframe has no attribute append"],
        "title": "Pandas DataFrame.append Removal & pd.concat Migration",
        "doc_content": """
OFFICIAL DOCS: pandas.concat
`DataFrame.append()` was deprecated in pandas 1.4 and completely REMOVED in pandas 2.0+.
Calling `df.append()` raises `AttributeError: 'DataFrame' object has no attribute 'append'`.

Recommended fixes:
- Replace `df.append(other_df)` with `pd.concat([df, other_df], ignore_index=True)`
- Replace loop appending to DataFrame with appending dicts to a list, then constructing DataFrame once:
  `records = []`
  `for item in items: records.append({'col': item})`
  `result_df = pd.DataFrame(records)`
""",
        "example_bad": "df = df.append(new_row, ignore_index=True)  # AttributeError in Pandas 2+",
        "example_good": "df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)"
    },
    {
        "id": "pandas_type_casting_missing",
        "category": "Pandas Type Conversion & Missing Values",
        "error_types": ["ValueError", "TypeError"],
        "keywords": ["astype", "float", "int", "nan", "null", "missing", "coerce", "fillna"],
        "title": "Pandas Type Conversion with Missing (NaN) Values",
        "doc_content": """
OFFICIAL DOCS: pandas.to_numeric / pandas.Series.astype
Converting columns with NaN or invalid strings like '$1,200.50' or 'N/A' directly with `.astype(int)` raises:
`ValueError: Cannot convert non-finite values (NA or inf) to integer` or `ValueError: Unable to parse string`.

Recommended fixes:
1. Strip currency symbols, commas, or whitespace first:
   `df['price'] = df['price'].astype(str).str.replace(r'[$,]', '', regex=True)`
2. Use `pd.to_numeric` with `errors='coerce'`:
   `df['price'] = pd.to_numeric(df['price'], errors='coerce')`
3. Handle missing values:
   `df['price'] = df['price'].fillna(df['price'].median())`
4. If nullable integer is needed:
   `df['count'] = df['count'].astype('Int64')  # Capital 'I' handles NA natively`
""",
        "example_bad": "df['sales'] = df['sales'].astype(int)  # Fails if NaNs exist",
        "example_good": "df['sales'] = pd.to_numeric(df['sales'], errors='coerce').fillna(0).astype(int)"
    },
    {
        "id": "pandas_pivot_table_duplicates",
        "category": "Pandas Pivoting & Reshaping",
        "error_types": ["ValueError", "KeyError"],
        "keywords": ["pivot", "pivot_table", "melt", "index contains duplicate entries"],
        "title": "Pandas pivot vs pivot_table Duplicate Index Handling",
        "doc_content": """
OFFICIAL DOCS: pandas.DataFrame.pivot / pandas.DataFrame.pivot_table
Calling `df.pivot(index='date', columns='category', values='sales')` raises
`ValueError: Index contains duplicate entries, cannot reshape` when there are multiple rows with the same index and column values.

Recommended fixes:
- Use `pivot_table` instead of `pivot`, which aggregates duplicates using `aggfunc`:
  `pivot_df = df.pivot_table(index='date', columns='category', values='sales', aggfunc='sum', fill_value=0).reset_index()`
- Always `.reset_index()` if you plan to use the index as a column for Plotly/Seaborn visualization.
""",
        "example_bad": "df.pivot(index='region', columns='product', values='sales')",
        "example_good": "df.pivot_table(index='region', columns='product', values='sales', aggfunc='sum', fill_value=0).reset_index()"
    },
    {
        "id": "plotly_express_column_requirements",
        "category": "Plotly Express Chart Generation",
        "error_types": ["ValueError", "KeyError", "TypeError"],
        "keywords": ["plotly", "px.bar", "px.line", "px.scatter", "px.pie", "px.histogram", "plotly.express"],
        "title": "Plotly Express Visualizations Best Practices & Error Prevention",
        "doc_content": """
OFFICIAL DOCS: plotly.express
Plotly Express expects tidy (long-form) DataFrames where columns specified in `x`, `y`, `color`, `hover_data` exist in `df.columns`.

Common errors:
1. `ValueError: Value of 'x' is not the name of a column in 'data_frame'`:
   Cause: After `groupby` or `pivot_table`, the grouping variable is in the index, not a column.
   Fix: Always run `.reset_index()` after aggregations before passing to Plotly!
2. `px.pie` issues with negative numbers:
   Cause: `values` in `px.pie` cannot be negative.
   Fix: Filter positive values or take absolute values: `df_pie = df[df['val'] > 0]`
3. Too many categories in `color` argument causing lag or unreadable legend:
   Fix: Group small categories into 'Other' or select top 10 categories.
4. Professional Styling:
   `fig.update_layout(template='plotly_white', title_x=0.5, font=dict(family='Inter, sans-serif'))`
   `fig.update_traces(textposition='outside')`
""",
        "example_bad": "grouped = df.groupby('category')['sales'].sum()\npx.bar(grouped, x='category', y='sales')  # KeyError: 'category'",
        "example_good": "grouped = df.groupby('category', as_index=False)['sales'].sum()\npx.bar(grouped, x='category', y='sales')"
    },
    {
        "id": "plotly_scatter_numeric_types",
        "category": "Plotly Express Scatter & Continuous Plots",
        "error_types": ["TypeError", "ValueError"],
        "keywords": ["scatter", "trendline", "continuous", "numeric", "size"],
        "title": "Plotly Scatter Plot Column Types and Trendline Requirements",
        "doc_content": """
OFFICIAL DOCS: plotly.express.scatter
Plotly scatter plots with `trendline='ols'` require `statsmodels` installed and both x and y columns to be numeric with no NaNs/Infs.
`size` argument must be strictly positive numeric values.

Recommended fixes:
- Ensure columns are clean floats:
  `plot_data = df.dropna(subset=['x_col', 'y_col']).copy()`
  `plot_data['x_col'] = pd.to_numeric(plot_data['x_col'], errors='coerce')`
  `plot_data['y_col'] = pd.to_numeric(plot_data['y_col'], errors='coerce')`
- If using `size`, replace negative or zero values:
  `plot_data['size_col'] = plot_data['size_col'].clip(lower=1)`
- Add trendline safely:
  `fig = px.scatter(plot_data, x='x_col', y='y_col', color='category', trendline=None)`
""",
        "example_bad": "px.scatter(df, x='age', y='income', size='profit')  # Fails if profit has negatives or NaNs",
        "example_good": "clean_df = df.dropna(subset=['age', 'income']).copy()\npx.scatter(clean_df, x='age', y='income')"
    },
    {
        "id": "python_attribute_error_dataframe",
        "category": "Python / Pandas Type and Attribute Errors",
        "error_types": ["AttributeError"],
        "keywords": ["attributeerror", "series has no attribute", "dataframe has no attribute", "str", "dt"],
        "title": "AttributeError Handling in Data Analysis Code",
        "doc_content": """
OFFICIAL DOCS: Python AttributeError in Pandas
Occurs when invoking a method on a Pandas object that belongs to a different type.

Common examples:
1. Calling DataFrame method on Series or vice-versa (e.g., `series.drop_duplicates(subset=...)` -> Series.drop_duplicates takes no `subset` argument).
2. Using `.str` on non-object / non-string series:
   `df['col'].astype(str).str.lower()`
3. Using `.dt` on string or object series instead of datetime:
   `pd.to_datetime(df['col']).dt.year`
4. Calling numpy/math functions on string columns.
""",
        "example_bad": "df['id'].drop_duplicates(subset=['id'])  # TypeError/AttributeError on Series",
        "example_good": "df.drop_duplicates(subset=['id'])"
    },
    {
        "id": "pandas_merge_join_key_mismatch",
        "category": "Pandas Merging & Joining",
        "error_types": ["MergeError", "KeyError", "ValueError"],
        "keywords": ["merge", "join", "mergeerror", "on", "left_on", "right_on", "keys"],
        "title": "Pandas Merge Key Mismatches and Type Discrepancies",
        "doc_content": """
OFFICIAL DOCS: pandas.merge
Merging two DataFrames raises `MergeError` or `ValueError` if the join keys have different data types
(e.g., merging an int64 ID column with an object/string ID column).

Recommended fixes:
1. Harmonize merge key types before merging:
   `df1['customer_id'] = df1['customer_id'].astype(str).str.strip()`
   `df2['customer_id'] = df2['customer_id'].astype(str).str.strip()`
   `merged = pd.merge(df1, df2, on='customer_id', how='left')`
2. Specify distinct column names if they differ:
   `merged = pd.merge(df1, df2, left_on='user_id', right_on='id', how='inner')`
3. Use suffixes for overlapping column names:
   `merged = pd.merge(df1, df2, on='id', suffixes=('_left', '_right'))`
""",
        "example_bad": "pd.merge(df1, df2, on='id')  # Fails if df1['id'] is int and df2['id'] is str",
        "example_good": "df1['id'] = df1['id'].astype(str)\ndf2['id'] = df2['id'].astype(str)\npd.merge(df1, df2, on='id', how='left')"
    },
    {
        "id": "numpy_division_by_zero_and_inf",
        "category": "Numerical Computation & Inf/NaN",
        "error_types": ["ZeroDivisionError", "FloatingPointError", "ValueError"],
        "keywords": ["zerodivisionerror", "division by zero", "inf", "nan", "divide"],
        "title": "Handling Division by Zero & Infinite Values in Feature Engineering",
        "doc_content": """
OFFICIAL DOCS: numpy.where / pandas.Series.replace
Dividing by zero in Pandas returns `np.inf` or `-np.inf`, which breaks subsequent Plotly visualizations
and correlation matrices (`ValueError: Out of range float values are not JSON compliant`).

Recommended fixes:
1. Use `np.where` with safe denominator check:
   `df['ratio'] = np.where(df['denominator'] != 0, df['numerator'] / df['denominator'], np.nan)`
2. Replace infinite values with NaN or 0:
   `df.replace([np.inf, -np.inf], np.nan, inplace=True)`
   `df['ratio'] = df['ratio'].fillna(0)`
""",
        "example_bad": "df['margin'] = df['profit'] / df['revenue']  # Produces inf if revenue == 0",
        "example_good": "df['margin'] = np.where(df['revenue'] > 0, df['profit'] / df['revenue'], 0)"
    },
    {
        "id": "plotly_treemap_sunburst_hierarchy",
        "category": "Plotly Hierarchical Visualizations (Treemap & Sunburst)",
        "error_types": ["ValueError", "KeyError"],
        "keywords": ["treemap", "sunburst", "path", "hierarchy", "values", "px.treemap", "px.sunburst"],
        "title": "Plotly Treemap & Sunburst Requirements & Non-Negative Values",
        "doc_content": """
OFFICIAL DOCS: plotly.express.treemap / plotly.express.sunburst
Treemaps and Sunburst charts decompose multi-level hierarchical dimensions.
Common errors:
1. `ValueError: Non-positive values found in 'values' column`:
   Cause: Treemap and Sunburst sectors represent proportions of area/angle; values cannot be zero or negative.
   Fix: Filter strictly positive values: `plot_df = df[df['value_col'] > 0].dropna(subset=path + ['value_col'])`
2. `ValueError: Value of 'path' must be a list of column names`:
   Fix: Provide a list of 2 or more categorical columns: `path=['Region', 'ProductCategory']`
3. NaN or missing values in path columns cause broken trees:
   Fix: `df[path] = df[path].fillna('Unassigned')`
""",
        "example_bad": "px.treemap(df, path='category', values='profit')  # Fails if path is string or profit has negatives",
        "example_good": "pos_df = df[df['profit'] > 0].dropna(subset=['region', 'category'])\npx.treemap(pos_df, path=['region', 'category'], values='profit')"
    },
    {
        "id": "plotly_dual_axis_pareto",
        "category": "Plotly Dual-Axis & Pareto Visualizations",
        "error_types": ["ValueError", "AttributeError"],
        "keywords": ["pareto", "secondary_y", "dual axis", "make_subplots", "cumulative"],
        "title": "Plotly Secondary Y-Axis & Pareto Analysis Best Practices",
        "doc_content": """
OFFICIAL DOCS: plotly.subplots.make_subplots / Secondary Y-Axis
Creating Pareto 80/20 charts requires combining bar traces (primary y-axis) with cumulative percentage line traces (secondary y-axis).

Recommended pattern:
1. Initialize subplots with secondary_y support:
   `from plotly.subplots import make_subplots`
   `fig = make_subplots(specs=[[{"secondary_y": True}]])`
2. Add primary bar trace:
   `fig.add_trace(go.Bar(x=df['cat'], y=df['val'], name='Volume'), secondary_y=False)`
3. Compute cumulative percentage and add line trace:
   `cum_pct = (df['val'].cumsum() / df['val'].sum()) * 100`
   `fig.add_trace(go.Scatter(x=df['cat'], y=cum_pct, mode='lines+markers', name='Cum %'), secondary_y=True)`
4. Set secondary axis range to [0, 105] to prevent line overflow:
   `fig.update_yaxes(range=[0, 105], secondary_y=True)`
""",
        "example_bad": "fig.add_trace(go.Scatter(...), secondary_y=True)  # Fails if make_subplots specs didn't enable secondary_y",
        "example_good": "fig = make_subplots(specs=[[{\"secondary_y\": True}]])\nfig.add_trace(go.Bar(...), secondary_y=False)\nfig.add_trace(go.Scatter(...), secondary_y=True)"
    },
    {
        "id": "plotly_violin_box_distribution",
        "category": "Plotly Distribution Plots (Box & Violin)",
        "error_types": ["ValueError", "TypeError"],
        "keywords": ["violin", "box", "distribution", "variance", "points", "px.violin", "px.box"],
        "title": "Plotly Box and Violin Plot Categorical Grouping",
        "doc_content": """
OFFICIAL DOCS: plotly.express.violin / plotly.express.box
Violin plots depict kernel density estimates along with interquartile boxes.
To compare variance across categories:
- Pass the categorical grouping variable to `x` and continuous metric to `y`:
  `px.violin(df, x='category', y='amount', color='category', box=True, points='outliers')`
- Set `box=True` inside `px.violin` so the median and IQR markers remain visible inside the violin density contours.
- Avoid grouping on high-cardinality categorical variables (>20 unique values) to prevent illegible axes.
""",
        "example_bad": "px.violin(df, x='id_column', y='revenue')  # High cardinality creates crowded unreadable chart",
        "example_good": "top_cats = df['category'].value_counts().head(8).index\nfiltered = df[df['category'].isin(top_cats)]\npx.violin(filtered, x='category', y='revenue', box=True)"
    }
]

