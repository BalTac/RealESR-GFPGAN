# Graphify Integration

> Sources: Graphify CLI & Antigravity Setup, 2026-11-03
> Raw: [Project Spec](../../raw/notes/project-spec.md)
> Updated: 2026-11-03

## Overview

Integrazione dello strumento **Graphify** (`graphifyy`) nel progetto per costruire automaticamente un knowledge graph strutturale di codice e documenti.

## Configurazione e Comandi

Lo strumento è installato nell'ambiente gestito tramite `uv` ed è registrato sia come skill che come regola/workflow:
- **Eseguibile**: `C:\Users\balta\.local\bin\graphify.exe`
- **Skill Workspace**: `.agents/skills/graphify/SKILL.md`
- **Regola Workspace**: `.agents/rules/graphify.md`
- **Workflow Workspace**: `.agents/workflows/graphify.md`

### Comandi Principali

```bash
# Costruzione iniziale del grafo sull'intero progetto
graphify .

# Aggiornamento incrementale (ri-estrazione solo file modificati)
graphify update .

# Interrogazione mirata del grafo per una domanda architetturale
graphify query "<domanda o concetto>"

# Ricerca del cammino più breve tra due entità/moduli
graphify path "<ModuloA>" "<ModuloB>"

# Spiegazione dettagliata di un nodo e delle sue dipendenze
graphify explain "<NomeComponente>"
```

## Output Generati (`graphify-out/`)
- `graph.json`: grafo in formato serializzato GraphRAG.
- `graph.html`: interfaccia visiva interattiva con mappa delle dipendenze.
- `GRAPH_REPORT.md`: sintesi leggibile delle comunità individuate e dei nodi cardine (*god nodes*).

## See Also
- [System Architecture](system-overview.md)
- [Documentation & Ingestion Workflow](../workflows/ingestion-workflow.md)
