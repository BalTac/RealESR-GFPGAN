# Documentation & Ingestion Workflow

> Sources: LLM-Wiki Methodology, 2026-11-03
> Raw: [Project Spec](../../raw/notes/project-spec.md)
> Updated: 2026-11-03

## Overview

Procedura standard per acquisire nuove informazioni, specifiche tecniche o codice e mantenerli allineati sia nella wiki persistente che nel grafo di conoscenza.

## Ciclo di Ingestione (Step-by-Step)

1. **Archiviazione Fonte Immutabile**:
   - Salvare il documento, schema o testo sorgente nella cartella appropriata dentro `raw/` (es. `raw/specs/feature-x.md`).
2. **Sintesi ed Estrazione Concetti**:
   - Creare o aggiornare l'articolo corrispondente in `wiki/<area>/<nome>.md`.
   - Includere la citazione della sorgente nel frontmatter `> Raw: [...]`.
3. **Aggiornamento Indice e Log**:
   - Aggiungere/aggiornare la voce in `wiki/index.md`.
   - Registrare l'operazione in `wiki/log.md` (append-only con data).
4. **Sincronizzazione Graphify**:
   - Eseguire `graphify update .` per integrare la nuova documentazione e i riferimenti nel knowledge graph navigabile.
5. **Verifica Evidenze (Audit)**:
   - Eseguire periodicamente `python .agents/skills/llm-wiki/scripts/check_evidence.py` per verificare che numeri, metriche e affermazioni cardine corrispondano alle fonti originali.

## See Also
- [System Architecture](../architecture/system-overview.md)
- [Graphify Integration](../architecture/graphify-integration.md)
