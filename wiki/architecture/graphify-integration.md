# Graphify Integration

> Sources: Graphify CLI & Antigravity Setup, 2026-11-03  
> Raw: [Project Spec](../../raw/notes/project-spec.md)  
> Updated: 2026-11-03  

## Overview

Integration of the **Graphify** tool (`graphifyy`) within the repository to automatically generate and maintain an AST-derived structural knowledge graph of code and technical documentation.

## Setup and Commands

The tool is integrated into the workspace environment and registered as a skill, workspace rule, and workflow:
- **Executable**: `C:\Users\balta\.local\bin\graphify.exe`
- **Workspace Skill**: `.agents/skills/graphify/SKILL.md`
- **Workspace Rule**: `.agents/rules/graphify.md`
- **Workspace Workflow**: `.agents/workflows/graphify.md`

### Core CLI Commands

```bash
# Full graph generation across the entire codebase
graphify .

# Incremental update (AST re-extraction for modified files only)
graphify update .

# Scoped query for architectural and code questions
graphify query "<question or concept>"

# Shortest dependency path between two modules or entities
graphify path "<ModuleA>" "<ModuleB>"

# Detailed concept breakdown and caller/callee inspection
graphify explain "<ComponentName>"
```

## Generated Outputs (`graphify-out/`)
- `graph.json`: Serialized GraphRAG knowledge graph.
- `graph.html`: Interactive visual graph browser displaying module clusters and dependencies.
- `GRAPH_REPORT.md`: Architectural summary highlighting community clusters and central entrypoints (*god nodes*).

## See Also
- [System Architecture](system-overview.md)
- [Documentation & Ingestion Workflow](../workflows/ingestion-workflow.md)
