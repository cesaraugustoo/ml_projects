# Scientific Discovery

A Python package for automating scientific discovery through multi-agent intelligent graph reasoning. This project builds upon and integrates approaches from two groundbreaking works in scientific discovery automation:

- [SciAgents: Automating scientific discovery through multi-agent intelligent graph reasoning](https://arxiv.org/abs/2409.05556) by Ghafarollahi and Buehler (2024)
- [Accelerating Scientific Discovery with Generative Knowledge Extraction, Graph-Based Representation, and Multimodal Intelligent Graph Reasoning](http://iopscience.iop.org/article/10.1088/2632-2153/ad7228) by Buehler (2024)

## Features

- Multi-agent system for scientific research automation
- Graph-based knowledge representation and reasoning
- Embedding-based concept analysis
- Integration with various LLM providers
- Comprehensive scientific workflow automation
- Advanced graph analysis and visualization tools

## Installation

### Basic Installation

```bash
pip install scientific-discovery
```

### Development Installation

```bash
git clone https://github.com/yourusername/scientific-discovery.git
cd scientific-discovery
pip install -e ".[dev]"
```

### Additional Features

For documentation tools:
```bash
pip install -e ".[docs]"
```

## Requirements

- Python >= 3.8
- See setup.py for complete dependency list

## Quick Start

```python
from scientific_discovery import (
    EmbeddingTools,
    GraphConfig,
    KnowledgeGraphBuilder,
    create_science_agent_group
)

# Initialize components
config = GraphConfig()
builder = KnowledgeGraphBuilder(config, output_dir="./output")
embedding_tools = EmbeddingTools()

# Create agent group
agent_group = create_science_agent_group(
    llm_client=your_llm_client,
    research_field="materials_science"
)

# Process research text
research_text = """
Novel materials for energy storage applications have garnered 
significant attention due to their potential in renewable energy systems.
"""

# Generate knowledge graph
graph, embeddings = builder.build_graph_from_text(
    research_text,
    generate_fn,
    "research_graph"
)

# Process with agent group
result = agent_group.process_research_task(research_text)
```

## Project Structure

```
scientific_discovery/
├── src/
│   ├── embedding_tools.py
│   ├── graph_gen.py
│   ├── graph_tools.py
│   ├── llm_tools.py
│   └── agent_tools/
│       ├── base.py
│       ├── science.py
│       └── utils.py
└── tests/
    ├── test_embedding.py
    ├── test_graph_generation.py
    └── ...
```

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=scientific_discovery

# Run specific test file
pytest tests/test_embedding.py
```

### Code Style

The project uses:
- Black for code formatting
- isort for import sorting
- mypy for type checking
- flake8 for linting

Run formatters:
```bash
black .
isort .
```

## Documentation

Generate documentation:
```bash
cd docs
make html
```

## Acknowledgments

This project is an integration and extension of methodologies presented in:

```bibtex
@article{ghafarollahi2024sciagentsautomatingscientificdiscovery,
      title={SciAgents: Automating scientific discovery through multi-agent intelligent graph reasoning}, 
      author={Alireza Ghafarollahi and Markus J. Buehler},
      year={2024},
      eprint={2409.05556},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2409.05556}, 
}

@article{buehler2024graphreasoning,
      author={Markus J. Buehler},
      title={Accelerating Scientific Discovery with Generative Knowledge Extraction, Graph-Based Representation, and Multimodal Intelligent Graph Reasoning},
      journal={Machine Learning: Science and Technology},
      year={2024},
      url={http://iopscience.iop.org/article/10.1088/2632-2153/ad7228},
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.