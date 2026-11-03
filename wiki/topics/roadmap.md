# Project Roadmap & Scope

> Sources: Project Team, 2026-11-03
> Raw: [Project Spec](../../raw/notes/project-spec.md)
> Updated: 2026-11-03

## Overview

Definizione degli obiettivi di alto livello del progetto (GFPGAN / Elaborazione immagini e modelli), piano di lavoro, requisiti operativi e fasi di sviluppo documentate progressivamente.

## Macro-Obiettivi
1. **Configurazione dell'Ambiente e Tooling**:
   - Adozione dell'infrastruttura di documentazione basata su **LLM-Wiki** e **Graphify**.
   - Gestione dell'interprete Python e delle dipendenze isolate tramite `uv`.
2. **Architettura e Modellazione**:
   - Mappatura completa dei moduli architetturali (GAN, generatori, prior facciali, loss functions).
   - Generazione di grafi di dipendenza e tracciamento delle relazioni con Graphify.
3. **Sviluppo ed Esperimenti**:
   - Logging continuo degli esperimenti e delle decisioni architetturali nella wiki.
   - Grounding delle decisioni tecniche con fonti salvate in `raw/`.

## See Also
- [System Architecture](../architecture/system-overview.md)
- [Graphify Integration](../architecture/graphify-integration.md)
