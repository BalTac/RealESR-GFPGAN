# Documentation & Ingestion Workflow

> Sources: LLM-Wiki Methodology, 2026-11-03  
> Raw: [Project Spec](../../raw/notes/project-spec.md)  
> Updated: 2026-11-03  

## Overview

Standard operating procedure for ingesting new technical documentation, research papers, and architectural changes while preserving synchronization across the persistent wiki and structural knowledge graph.

## Ingestion Lifecycle (Step-by-Step)

1. **Immutable Source Storage**:
   - Save the raw reference document, benchmark report, or specification in the appropriate subfolder within `raw/` (e.g., `raw/specs/feature-x.md`).
2. **Concept Synthesis & Structuring**:
   - Create or update the corresponding article in `wiki/<domain>/<title>.md`.
   - Provide the source provenance header `> Raw: [...]`.
3. **Index & Log Synchronization**:
   - Update entries in `wiki/index.md`.
   - Record the operation chronologically in `wiki/log.md` (append-only).
4. **Graphify Synchronization**:
   - Run `graphify update .` to index new documentation nodes, links, and relationships.
5. **Evidence Audit**:
   - Periodically execute evidence checks to ensure all metrics, formulas, and statements remain grounded in source files.

## See Also
- [System Architecture](../architecture/system-overview.md)
- [Graphify Integration](../architecture/graphify-integration.md)
