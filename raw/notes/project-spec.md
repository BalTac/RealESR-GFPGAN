# Project Specification & Architecture Notes

Data: 2026-11-03
Autore: Balta / DeepMind Pair Programming Session

## Requisiti e Obiettivi
1. Adozione congiunta della skill `llm-wiki` e della skill / tool `graphifyy`.
2. Costruzione di una documentazione modulare, persistente ed esaustiva per il progetto GFPGAN.
3. Mantenimento del tracciamento di dipendenze (AST code parsing e graph query) integrato con una base di conoscenza ad articoli Markdown sintetizzati.
4. Mantenimento della separazione tra sorgenti immutabili (`raw/`), base di conoscenza compilata (`wiki/`) e grafo di interconnessione (`graphify-out/`).
