# System Architecture

> Sources: Project Team, 2026-11-03  
> Raw: [Project Spec](../../raw/notes/project-spec.md)  
> Updated: 2026-11-03  

## Overview

Structural overview of the project layers and the unified workflow connecting code, technical documentation, and structural knowledge graphs.

## Integration Diagram

```
                        +---------------------------+
                        |      Input / Sources      |
                        | (Documents, Papers, Code) |
                        +-------------+-------------+
                                      |
                       +--------------+--------------+
                       |                             |
                       v                             v
           +-----------------------+     +-----------------------+
           |       LLM-Wiki        |     |       Graphify        |
           |  (Knowledge / Docs)   |     | (AST Graph & Schema)  |
           |                       |     |                       |
           | • raw/ (Sources)      |     | • graph.json          |
           | • wiki/ (Synthesis)   |     | • graph.html          |
           | • index.md & log.md   |     | • GRAPH_REPORT.md     |
           +-----------+-----------+     +-----------+-----------+
                       |                             |
                       +--------------+--------------+
                                      |
                                      v
                        +---------------------------+
                        |     AI Agent & User       |
                        |  Navigation & Development |
                        +---------------------------+
```

## Key Components
- **Raw Sources (`raw/`)**: Preserves technical specifications, benchmark logs, and foundational research papers without modification.
- **Synthesized Wiki (`wiki/`)**: Interconnected, human- and machine-readable Markdown documentation articles.
- **Knowledge Graph (`graphify-out/`)**: Graph representation mapping code dependencies, module calls, and structural impact analysis.
- **Unified Restoration Engine (`realesr_gfpgan/`)**: Standalone hybrid restoration pipeline integrating RRDBNet super-resolution with calibrated generative facial priors and seamless alpha-mask compositing.

## See Also
- [Project Roadmap & Scope](../topics/roadmap.md)
- [Graphify Integration](graphify-integration.md)
