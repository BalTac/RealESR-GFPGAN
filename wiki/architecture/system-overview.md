# System Architecture

> Sources: Project Team, 2026-11-03
> Raw: [Project Spec](../../raw/notes/project-spec.md)
> Updated: 2026-11-03

## Overview

Panoramica strutturale dei layer del progetto e del flusso di lavoro combinato tra codice, documentazione e grafo di conoscenza.

## Schema di Integrazione

```
                        +---------------------------+
                        |      Input / Sources      |
                        | (Documenti, Papers, Code) |
                        +-------------+-------------+
                                      |
                       +--------------+--------------+
                       |                             |
                       v                             v
           +-----------------------+     +-----------------------+
           |       LLM-Wiki        |     |       Graphify        |
           |  (Conoscenza/Doc)     |     | (Grafo AST & Schema)  |
           |                       |     |                       |
           | • raw/ (Fonti)        |     | • graph.json          |
           | • wiki/ (Sintesi)     |     | • graph.html          |
           | • index.md & log.md   |     | • GRAPH_REPORT.md     |
           +-----------+-----------+     +-----------+-----------+
                       |                             |
                       +--------------+--------------+
                                      |
                                      v
                        +---------------------------+
                        |     AI Agent & User       |
                        | Navigazione e Sviluppo    |
                        +---------------------------+
```

## Componenti Chiave
- **Sorgenti Raw (`raw/`)**: Conservano le specifiche, i paper e i log di sessione senza alterazioni.
- **Wiki Sintetica (`wiki/`)**: Documenti in markdown interconnessi consultabili sia dall'assistente che dall'utente.
- **Knowledge Graph (`graphify-out/`)**: Risolve istantaneamente le relazioni strutturali, le chiamate tra funzioni e l'impatto delle modifiche.

## See Also
- [Project Roadmap & Scope](../topics/roadmap.md)
- [Graphify Integration](graphify-integration.md)
