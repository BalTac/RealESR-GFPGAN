# Wiki Schema & Governance

> Sources: Project Governance, 2026-11-03
> Raw: [Project Spec](../raw/notes/project-spec.md)
> Updated: 2026-11-03

Linee guida per la gestione della documentazione e della base di conoscenza del progetto.

## Architettura a 3 Livelli

1. **Layer 1: `raw/` (Fonti Immutabili)**
   - Contiene file sorgente originali, specifiche, note di requisiti, paper scientifici, trascrizioni o documentazione esterna.
   - Non viene mai modificato direttamente dopo l'ingestione.
   - Organizzato in sottocartelle tematiche (es. `raw/specs/`, `raw/papers/`, `raw/notes/`).

2. **Layer 2: `wiki/` (Base di Conoscenza Sintetizzata)**
   - Articoli Markdown strutturati, sintetizzati e curati dall'LLM.
   - Organizzati per sottocartelle (es. `topics/`, `architecture/`, `workflows/`).
   - Ogni articolo contiene metadati (`Title`, `Raw Sources`, `Updated`, `Tags`) e wikilink bidirezionali.
   - `wiki/index.md` mantiene il catalogo globale.
   - `wiki/log.md` traccia in modo append-only ogni aggiornamento.

3. **Layer 3: `graphify-out/` (Knowledge Graph)**
   - Generato e mantenuto da Graphify.
   - `graph.json`: grafo deterministico interrogabile (nodi, archi, comunità, god-nodes).
   - `graph.html`: visualizzazione interattiva dei componenti e delle dipendenze.
   - `GRAPH_REPORT.md`: report testuale delle connessioni architetturali.

## Invariante di Grounding
Ogni fatto, numero, metrica o decisione critica riportata nella wiki deve essere rintracciabile nei file sorgente in `raw/` o nel codice tracciato da Graphify.
Lo script di verifica in `.agents/skills/llm-wiki/scripts/check_evidence.py` può essere eseguito per audit di coerenza.
