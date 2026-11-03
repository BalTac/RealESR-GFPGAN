# Wiki Log

Append-only log delle modifiche e delle operazioni eseguite sulla Knowledge Base.

## 2026-11-03
- **INIT**: Inizializzazione della base di conoscenza LLM-Wiki e configurazione di Graphify CLI (`graphifyy`).
- **CONFIG**: Aggiunte le skills `.agents/skills/llm-wiki` e `.agents/skills/graphify` nel workspace.
- **DOCS**: Creazione dell'indice iniziale, delle sezioni di architettura e della guida ai flussi di documentazione.
- **INGEST**: Acquisizione del post di ricerca "Sparks of Weight-Space Programming" (Rikiya Takehi et al., 2026-11-02) salvato in `raw/weight-space-programming/weight-space-programming.md`. Compilato articolo in `wiki/topics/weight-space-programming.md` e aggiornato `wiki/index.md`.
- **EXPERIMENT**: Eseguito benchmark inferenza baseline su 3 immagini di test (`Blake_Lively`, `10045`, `00`). Eseguita calibrazione multi-pass per rimozione pelle waxy/plasticosa via micro-texture injection e pulizia del background con filtro subpixel Lanczos-4. Generata pagina HTML comparativa in `results/comparison.html`.
- **TEST & UI**: Elaborata nuova immagine utente `Foto_3.jpg` (Baseline: 45.41s, Calibrated: 46.42s). Aggiornata `results/comparison.html` con layout affiancato a 3 colonne, slider interattivi prima/dopo e visualizzazione Fullscreen con slider attivo.
