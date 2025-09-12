# Streamlit Cheat Sheet

**Sources:** [Streamlit documentation](https://docs.streamlit.io/#), (Streamlit API cheat sheet)[https://docs.streamlit.io/develop/quick-reference/cheat-sheet]

**Description:**  
Streamlit is an open-source Python framework for creating dynamic data apps. It simplifies creating, displaying, and sharing apps for Data, AI, and ML engineers.

---

## Streamlit Syntax

| Syntax | Description | Example |
|--------|-------------|---------|
| `st.function(args)` | Display content or create a widget | `st.write("Hello")` <br> `st.dataframe(df)` <br> `st.image("image.png")` <br> `st.slider("Pick a number",0,10)` |
| With context manager: `with st.<container>(...)` | Organize layout, group widgets, enable nested behavior | ```python<br>with st.container():<br> &nbsp;&nbsp;st.write("Inside container")<br> &nbsp;&nbsp;st.line_chart(df)<br>``` |
| Decorators: `@st.<decorator>(...)` | Define fragments, dialogs, or cached computations | ```python<br>@st.fragment<br>def my_function():<br> &nbsp;&nbsp;st.line_chart(df)<br> &nbsp;&nbsp;st.button("Update")<br>``` |
| Object-oriented style | Update previously displayed elements dynamically | ```python<br>element = st.empty()<br>element.text("Hello")<br>bar = st.progress(50)<br>bar.progress(100)<br>``` |
| Caching | Cache results of computations or persistent resources | ```python<br>@st.cache_data<br>def expensive_computation(...):<br> &nbsp;&nbsp;return result<br>@st.cache_resource<br>def create_session(...):<br> &nbsp;&nbsp;return session<br>``` |

---

## Essentials to Know

| Category | Command | What it does |
|----------|---------|--------------|
| Install & Import | `pip install streamlit` | Install Streamlit |
|  | `streamlit run first_app.py` | Run a Streamlit app |
|  | `import streamlit as st` | Import Streamlit |
| Pre-release features | `pip uninstall streamlit` | Uninstall Streamlit |
|  | `pip install streamlit-nightly --upgrade` | Install nightly build (experimental) |
| Command line | `streamlit cache clear` | Clear cached data |
|  | `streamlit config show` | Show config options |
|  | `streamlit docs` | Open Streamlit docs |
|  | `streamlit hello` | Run demo app |
|  | `streamlit help` | Show help info |
|  | `streamlit init` | Initialize project |
|  | `streamlit run streamlit_app.py` | Run specified app |
|  | `streamlit version` | Show Streamlit version |

---

## Magic Commands

| Command | Description |
|---------|-------------|
| `"This is some **Markdown**"` | Implicitly calls `st.write()` |
| `my_variable` | Display value with `st.write()` |
| `"dataframe:", my_data_frame` | Display dataframe inline |

---

## Display Text

| Command | Description |
|---------|-------------|
| `st.write("Most objects")` | Display any object (df, list, func) |
| `st.text("Fixed width text")` | Fixed-width text |
| `st.markdown("_Markdown_")` | Markdown text |
| `st.latex(r""" e^{i\pi} + 1 = 0 """)` | Render LaTeX |
| `st.title("My title")` | Display title |
| `st.header("My header")` | Display header |
| `st.subheader("My sub")` | Display subheader |
| `st.code("for i in range(8): foo()")` | Display code block |
| `st.badge("New")` | Show badge |
| `st.html("<p>Hi!</p>")` | Render raw HTML |

---

## Display Data

| Command | Description |
|---------|-------------|
| `st.dataframe(my_dataframe)` | Interactive Dataframe |
| `st.table(data.iloc[0:10])` | Static Table |
| `st.json({"foo":"bar"})` | Show JSON |
| `st.metric("My metric",42,2)` | Display metric with delta |

---

## Display Media

| Command | Description |
|---------|-------------|
| `st.image("./header.png")` | Show image |
| `st.logo("logo.jpg")` | Show logo |
| `st.audio(data)` | Play audio |
| `st.video(data)` | Play video |
| `st.video(data, subtitles="./subs.vtt")` | Play video with subtitles |

---

## Display Charts

| Command | Description |
|---------|-------------|
| `st.area_chart(df)` | Area chart |
| `st.bar_chart(df)` | Bar chart |
| `st.line_chart(df)` | Line chart |
| `st.map(df)` | Map visualization |
| `st.scatter_chart(df)` | Scatter chart |
| `st.altair_chart(chart)` | Altair chart |
| `st.plotly_chart(fig)` | Plotly chart |
| `st.pydeck_chart(chart)` | PyDeck chart |
| `st.pyplot(fig)` | Matplotlib chart |
| `st.graphviz_chart(fig)` | Graphviz chart |
| `st.vega_lite_chart(df,spec)` | Vega-Lite chart |

---

## Sidebar, Tabs & Layout

| Feature | Example |
|---------|---------|
| Sidebar radio | `st.sidebar.radio("Select one:", [1,2])` |
| Sidebar with context | `with st.sidebar: st.radio("Select one:", [1,2])` |
| Columns | `col1, col2 = st.columns(2)` |
| Tabs | `tab1, tab2 = st.tabs(["Tab 1","Tab 2"])` |
| Expandable | `expand = st.expander("My label")` |
| Popover | `pop = st.popover("Button label")` |

---

## Control Flow

| Command | Description |
|---------|-------------|
| `st.stop()` | Stop execution |
| `st.rerun()` | Rerun script |
| `st.switch_page("pages/my_page.py")` | Navigate to another page |
| `with st.form(key="my_form"):` | Group widgets in a form |
| `@st.dialog("Welcome!")` | Define modal dialog |
| `@st.fragment` | Define fragment function |

---

## Widgets

| Command | Description |
|---------|-------------|
| `st.button("Click me")` | Button |
| `st.checkbox("I agree")` | Checkbox |
| `st.radio("Pick one", ["cats","dogs"])` | Radio select |
| `st.selectbox("Pick one", ["cats","dogs"])` | Dropdown select |
| `st.multiselect("Buy", ["milk","apples"])` | Multi-select |
| `st.slider("Pick a number", 0, 100)` | Slider |
| `st.text_input("First name")` | Text input |
| `st.date_input("Your birthday")` | Date picker |
| `st.file_uploader("Upload a CSV")` | File upload |
| `st.color_picker("Pick a color")` | Color picker |

---

## Chat Apps

| Command | Description |
|---------|-------------|
| `with st.chat_message("user"):` | Insert chat message |
| `st.chat_input("Say something")` | Chat input widget |

---

## Placeholders, Layout & Mutate Data

| Command | Description |
|---------|-------------|
| `element = st.empty()` | Placeholder element |
| `elements = st.container()` | Container for multiple elements |
| `flex = st.container(horizontal=True)` | Horizontal flex layout |
| `element.add_rows(df2)` | Add rows to chart/dataframe |

---

## Code Display

| Command | Description |
|---------|-------------|
| `with st.echo():` | Display code while executing |

---

## Help & Options

| Command | Description |
|---------|-------------|
| `st.help(pandas.DataFrame)` | Show docs for object |
| `st.set_page_config(layout="wide")` | Configure page layout |

---

## Data Connections

| Command | Description |
|---------|-------------|
| `st.connection("pets_db", type="sql")` | Connect to DB |
| `conn.query(query)` | Execute query |

---

## Caching

| Command | Description |
|---------|-------------|
| `@st.cache_data` | Cache data objects |
| `@st.cache_resource` | Cache global resources |

---

## Progress & Status

| Command | Description |
|---------|-------------|
| `with st.spinner("In progress"):` | Show spinner |
| `bar = st.progress(50)` | Show/update progress bar |
| `st.balloons()` | Show balloons |
| `st.snow()` | Show snow |
| `st.toast("Warming up")` | Show toast |
| `st.error("Error")` | Error message |
| `st.warning("Warning")` | Warning message |
| `st.success("Success")` | Success message |

---

## Personalization & Users

| Command | Description |
|---------|-------------|
| `if not st.user.is_logged_in:` | Check login |
| `st.login("provider")` | Log in user |
| `st.logout()` | Log out |
| `st.context.cookies` | Access cookies |
| `st.context.headers` | Access headers |

