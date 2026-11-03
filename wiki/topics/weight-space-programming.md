# Sparks of Weight-Space Programming

> Sources: Rikiya Takehi, Jayden Teoh, Alex Zhang, Omar Khattab, 2026-11-02
> Raw: [Weight-Space Programming Blog Post](../../raw/weight-space-programming/weight-space-programming.md)
> Updated: 2026-11-03

## Overview

Il concetto di **Weight-Space Programming** descrive la capacità emergente dei Large Language Model di frontiera (nello studio viene testato GPT-6 Astra) di leggere, interpretare, correggere e scrivere direttamente i pesi continui delle reti neurali, senza affidarsi alla discesa del gradiente o a cicli di addestramento tradizionali. Tramite puro ragionamento simbolico e geometrico sui tensori dei pesi (o con un budget limitato di passaggi in avanti in un kernel Python), il modello è in grado di identificare architetture e task, diagnosticare fallimenti funzionali, generare policy da zero e applicare patch chirurgiche che modificano frazioni minime dei parametri (ad es. lo 0.016% dei pesi in Qwen3.5-0.8B) risolvendo loop di ragionamento ed elevando drasticamente le prestazioni.

## Capacità di Lettura dei Pesi (Reading)

Gli esperimenti dimostrano che un LLM può dedurre il comportamento funzionale di un modello esaminando unicamente i valori numerici dei pesi:

1. **Riconoscimento del Task da Pesi Policy (Esperimento 1)**:
   - Architettura: MLP 500 → 32 → 32 → 5 con attivazione tanh (17,253 weights).
   - Addestramento: PPO per 1 billion steps su quattro giochi su griglia 10×10 (Breakout, Pong, Pac-Man, Freeway).
   - Risultato: Astra identifica con precisione il gioco corrispondente a ciascuna rete (es. Breakout al 99% di confidenza) analizzando dove si concentrano i pesi del primo strato (es. la paletta sul bordo inferiore) e come l'output separa le azioni.

2. **Decodifica di Modelli Diffusivi (Esperimento 2)**:
   - Architettura: MLP 160 → 64 → 64 → 144 con ReLU (23,824 weights).
   - Task: DDPM per immagini binarie 12×12 di cerchi, quadrati e triangoli pieni.
   - Risultato: Astra individua la forma geometrica generata esaminando i vettori dei pesi dello strato finale, disegnando i contorni delle figure senza calcolare alcun forward pass.

3. **Ispezione dei Blocchi Intermedi nei Modelli di Linguaggio (Esperimento 3)**:
   - Modelli: Transformer a livello di carattere (4 layers × 6 heads, width 384, 7.16 million weights) addestrati su 1 billion characters di codice Python, letteratura giapponese in romaji e inglese da Wikipedia.
   - Condizioni di test: Strati di embedding e di output completamente rimossi; Astra analizza solo i quattro blocchi intermedi con script Python (senza mai eseguire la rete su input).
   - Risultato: Identifica correttamente la lingua e localizza esattamente le teste di copiatura (induction heads) — layer.head 2.1, 3.1, 3.5 per Python, 3.3 per Romaji, 3.1 per English — corrispondenti ai valori misurati sul modello (score 0.22 to 0.55 per copying heads vs at most 0.02 per le altre).

## Capacità di Scrittura e Correzione dei Pesi (Writing & Fixing)

1. **Riparazione di Policy in Minimi Locali (Esperimento 4)**:
   - Task: Livello car-on-ramp nell'ambiente fisico Kinetix (171,664 weights, 6 body parts, 16 azioni motore/thruster).
   - La policy PPO originale converge in un minimo locale (stalla sulla rampa, 0% di successo in 256 episodi).
   - Intervento: Astra modifica il 3.5% of the weights in un singolo tentativo senza tool, inserendo un percorso separato che legge posizione e velocità, fa retrocedere l'auto per accumulare accelerazione e disattiva il bypass una volta superata la striscia rossa.
   - Risultato: 100% success su tutti i 1,024 episodes.

2. **Generazione di Pesi da Zero (Esperimento 5 - MinAtar)**:
   - Architettura: 600 → 32 → 32 → 6 (tanh), pesi inizializzati a 0.
   - Senza dati, gradienti o rollouts d'ambiente, Astra scrive i pesi direttamente in testo partendo dalle regole e dal codice dei giochi:
     - **Breakout**: 517.4 (Astra) vs 25.3 (PPO, 500M steps) e 15.8 (PPO, 100M steps).
     - **Asterix**: 53.2 (Astra) vs 55.2 (PPO, 500M steps) e 25.8 (PPO, 100M steps).
     - **Seaquest**: 53.1 (Astra) vs 88.9 (PPO, 500M steps) e 5.6 (PPO, 100M steps).

## Scenari su Scala Reale e Tool-Use

1. **Sintesi di una Rete per il Gioco del Go (Esperimento 6)**:
   - Architettura: 243 → 512 × 6 → 82 con ReLU (1,480,274 weights) su scacchiera Go 9×9.
   - Astra programma la rete livello per livello in uno script Python di 336 righe: individua le libertà delle pietre, propaga le catene connesse e calcola il punteggio della mossa.
   - Con 50 forward passes di verifica, la rete generata batte la rete PPO di pari dimensione addestrata su 20 million games.

2. **Rilevamento di Modelli Sicofantici (Esperimento 7)**:
   - Modello: Due copie di Gemma 3 270M (18 layers, token embedding congelati), una onesta (segue 1–3% delle risposte errate dell'utente) e una sicofantica (segue il 100% delle risposte errate).
   - Con l'ausilio di un tool per analizzare i pesi di attenzione su testi generati da Astra, il modello isola come la testa 10.2 nella rete sicofantica concentri l'attenzione all'85.7% sull'ipotesi dell'utente anziché sui numeri del problema (0.75% nel modello onesto).

3. **Ottimizzazione Chirurgica di Ragionamento Matematico (Esperimento 8 - Qwen3.5-0.8B)**:
   - Modello: Qwen3.5-0.8B (24 layers, 0.8 billion weights).
   - Problema diagnosticato: Tendenza del modello a cadere in loop di pensiero infiniti (il 93% delle risposte su MATH-500 toccava il limite di 32,768 tokens).
   - Risoluzione: Astra individua una singola direzione nei blocchi MLP dei layer 15 e 19 responsabile del loop e la sottrae da 64 colonne di output per ciascuno.
   - Impatto: Modifica di appena 131,072 pesi (pari allo 0.016% del totale), senza alcun retraining con discesa del gradiente:
     - **MATH-500**: da 7.9% a 48.1% (con 50 forward passes).
     - **GSM8K**: da 13.8% a 67.1%.
     - **IFEval** (istruzioni generali): da 21.0% a 62.3%.
     - Tasso di risposte che toccano il limite di token (Hit token limit): sceso da 93% a 33% (93.0 contro 33.1).
   - Limiti osservati su Gemma 3 270M: Su un modello che non soffre di loop (solo 8.5% di risposte al limite, contro il 93% di Qwen), l'editing dei pesi non aggiunge capacità mancanti (MATH-500 passa da 6.8 a 7.7 con forward passes), evidenziando che l'approccio sblocca capacità latenti mascherate da difetti strutturali, ma non crea conoscenza ex novo.

## Implicazioni Future
- **Audit senza inferenza**: Ispezione di backdoor, bias o allineamento direttamente dai checkpoint.
- **Riduzione drastica dei costi di ottimizzazione**: Inizializzazione o patch one-shot senza costosi cicli di data curation e addestramento.
- **Unlearning e Steering a zero dati**: Disattivazione selettiva di comportamenti indesiderati preservando il resto della rete e prevenendo il catastrophic forgetting.
- **Model Merging Architetturale**: Innesto (*grafting*) di circuiti funzionali tra modelli eterogenei senza dover mediare semplicemente i pesi di modelli identici.

## See Also
- [Project Roadmap & Scope](roadmap.md)
- [System Architecture](../architecture/system-overview.md)
- [Documentation & Ingestion Workflow](../workflows/ingestion-workflow.md)
