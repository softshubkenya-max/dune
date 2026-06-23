# DUNE Modules - Quick Reference

All integrated repositories are now available as Python modules in DUNE.

## Using the Modules

### List All Modules
```python
from dune import list_modules

modules = list_modules()
for name, info in modules['modules'].items():
    print(f"{info['name']}: {info['description']}")
```

### Get a Specific Module
```python
from dune import get_module

# RAG module
rag = get_module('haystack')
rag.index_documents(documents)
results = rag.retrieve("your query")

# Semantic Kernel
orchestrator = get_module('semantic_kernel')
result = orchestrator.orchestrate(workflow)

# Botpress
bot = get_module('botpress')
response = bot.process_message("Hello!")

# KaTeX Math Rendering
math = get_module('katex')
html = math.render(r"\frac{x^2}{y}")
svg = math.render_svg(r"\sum_{i=1}^{n} x_i")

# Markdown Processing
md = get_module('markdown')
processed = md.process(markdown_content)

# Data Visualization
vis = get_module('visualization')
chart = vis.visualize(data, chart_type="line")

# Generative UI
ui = get_module('generative_ui')
component = ui.get_component("chat_interface")
pattern = ui.get_pattern("streaming_response")
```

### Check Module Status
```python
from dune import get_module_status

status = get_module_status()
# Returns status for all modules
```

### Using the Modules Manager Directly
```python
from dune import get_modules_manager

manager = get_modules_manager()

# List modules by category
retrieval = manager.list_by_category('retrieval')
orchestration = manager.list_by_category('orchestration')
ui = manager.list_by_category('ui')

# Get all capabilities
capabilities = manager.get_capabilities()

# Get complete status
status = manager.get_status()
```

## Available Modules

### 1. Haystack (RAG)
**Module:** `haystack`
**Category:** retrieval
**Functions:**
- `index_documents(docs)` - Index documents for retrieval
- `retrieve(query, top_k)` - Search for relevant documents
- `generate(query, context)` - Generate answers

### 2. Semantic Kernel
**Module:** `semantic_kernel`
**Category:** orchestration
**Functions:**
- `add_plugin(name, code)` - Add plugins
- `orchestrate(workflow)` - Execute workflows
- `manage_context(key, value)` - Manage execution context
- `get_status()` - Get kernel status

### 3. Botpress
**Module:** `botpress`
**Category:** conversational_ai
**Functions:**
- `process_message(msg, session)` - Process user message
- `generate_response(context)` - Generate bot response
- `manage_dialog(session, state)` - Manage dialog state
- `configure_nlu(config)` - Configure NLU
- `add_integration(type, config)` - Add channel integrations

### 4. KaTeX
**Module:** `katex`
**Category:** math
**Functions:**
- `render(latex, options)` - Render LaTeX to HTML
- `render_svg(latex, options)` - Render LaTeX to SVG
- `parse_math(content, delimiter)` - Find math in text
- `validate_latex(latex)` - Validate LaTeX syntax
- `batch_render(expressions, format)` - Render multiple expressions

### 5. LLM-Markdown
**Module:** `markdown`
**Category:** text_processing
**Functions:**
- `process(content)` - Process markdown with LLM calls
- `embed_llm_calls(content, queries)` - Add LLM calls to markdown
- `generate_from_template(template, context)` - Generate from template
- `parse_frontmatter(content)` - Extract YAML frontmatter
- `extract_headings(content)` - Extract heading hierarchy

### 6. GPT-Vis
**Module:** `visualization`
**Category:** visualization
**Functions:**
- `visualize(data, chart_type, options)` - Generate visualization
- `analyze_visual(data)` - Suggest visualizations
- `generate_insights(data)` - Generate insights
- `compare_datasets(data1, data2)` - Compare datasets
- `export_visualization(spec, format)` - Export visualization

### 7. Generative UI
**Module:** `generative_ui`
**Category:** ui
**Functions:**
- `get_component(type)` - Get UI component
- `get_pattern(name)` - Get design pattern
- `list_components()` - List all components
- `list_patterns()` - List all patterns
- `register_custom_component(name, spec)` - Register custom component
- `set_theme(theme)` - Set UI theme
- `get_responsive_config(breakpoint)` - Get responsive config
- `get_accessibility_config()` - Get a11y config

## Module Categories

### By Category
- **retrieval**: haystack
- **orchestration**: semantic_kernel
- **conversational_ai**: botpress
- **math**: katex
- **text_processing**: markdown
- **visualization**: visualization
- **ui**: generative_ui

## Integration Points

All modules are integrated with:
1. **DUNE Core**: Can be used with DUNE orchestrator
2. **Web UI**: Accessible via `/api/` endpoints
3. **CLI**: Available in Python REPL
4. **Plugins**: Can extend other modules

## Example: Complete Workflow

```python
from dune import get_module, create_dune

# Initialize DUNE
dune = create_dune()

# Get modules
rag = get_module('haystack')
orchestrator = get_module('semantic_kernel')
vis = get_module('visualization')
md = get_module('markdown')
ui = get_module('generative_ui')

# RAG retrieval
rag.index_documents(documents)
context = rag.retrieve("What is quantum computing?")

# Orchestrate workflow
workflow = {
    "name": "analysis",
    "steps": [
        {"name": "retrieve", "type": "rag"},
        {"name": "reason", "type": "logic"},
        {"name": "visualize", "type": "chart"}
    ]
}
result = orchestrator.orchestrate(workflow)

# Visualize results
chart = vis.visualize(result, chart_type="bar")

# Generate markdown report
md_template = "# Analysis Report\n\n{{ content }}"
report = md.generate_from_template(md_template, {"content": result})

# Get UI for interaction
chat_ui = ui.get_component("chat_interface")
streaming_pattern = ui.get_pattern("streaming_response")

# Use DUNE for reasoning
explanation = dune.explain(result)
```

## Troubleshooting

### Module not available
```python
from dune import get_modules_manager

manager = get_modules_manager()
status = manager.get_status()
# Check 'available' field for each module
```

### Missing dependencies
Some modules require external packages:
- `haystack`: Requires Haystack package
- `semantic_kernel`: Requires Semantic Kernel package

Install with:
```bash
pip install haystack semantic-kernel botpress-api
```

### API Configuration
Some modules need API keys:
```python
# Semantic Kernel with OpenAI
sk = get_module('semantic_kernel')
# Requires OPENAI_API_KEY environment variable
```

## Module Architecture

```
dune/
├── modules.py                    # Unified manager
├── modules_haystack.py           # RAG adapter
├── modules_semantic_kernel.py    # Orchestration adapter
├── modules_botpress.py           # Conversational AI adapter
├── modules_katex.py              # Math rendering adapter
├── modules_markdown.py           # Markdown adapter
├── modules_gptvis.py             # Visualization adapter
├── modules_generative_ui.py      # UI adapter
└── integrations.py               # Capability layer
```
