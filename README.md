# Premier League 2020–2021 | Data Mining e classificazione dei risultati

**Progetto universitario di Data Mining** · Analisi degli eventi calcistici · Python · Pandas · Analisi esplorativa dei dati · Feature Engineering · XGBoost

> **Il progetto in breve.** Abbiamo analizzato **607.656 eventi di gioco** (34 colonne nel dataset originale) relativi alla stagione 2020–2021 della Premier League inglese, ricostruito **380 partite** e creato **760 osservazioni squadra–partita** per le analisi e la classificazione dei risultati. Il notebook originale riporta un'**accuracy sul test del 78,29%** e un **F1 macro di 0,7920** per un classificatore XGBoost a due stadi. Si tratta di **risultati salvati nel notebook originale**, non di metriche ricalcolate per questa documentazione.

## Obiettivo

Il dataset di partenza contiene eventi di gioco e non dispone di un identificativo di partita già pronto all'uso. Il progetto trasforma gli eventi individuali in statistiche strutturate per ciascuna squadra e partita, esplora le prestazioni di squadre e giocatori e analizza la possibilità di classificare i risultati come **vittoria, pareggio o sconfitta** utilizzando le statistiche dell'intera partita.

## Metodologia

1. **Esplorazione e pulizia dei dati:** analisi delle colonne e associazione degli identificativi delle squadre ai rispettivi nomi.
2. **Ricostruzione delle partite:** utilizzo degli eventi `Start` del `FirstHalf` per ricostruire le 380 partite e assegnare un `match_id` agli eventi.
3. **Creazione delle variabili (feature engineering):** aggregazione di passaggi, tiri, intercetti, respinte, contrasti, falli, cartellini, recuperi palla e coordinate spaziali; gestione degli autogol e confronto con le statistiche della squadra avversaria.
4. **Analisi esplorativa:** studio dei risultati, dei tiri, dei passaggi, delle correlazioni, della classifica, dei marcatori, degli assist e delle prestazioni di difensori e portieri, con grafici e tabelle.
5. **Classificazione dei risultati:** addestramento di due modelli XGBoost in sequenza: il primo distingue **pareggio e non pareggio**, il secondo distingue **vittoria e sconfitta** nei casi classificati come non pareggio. Sono previsti una ricerca casuale degli iperparametri e la scelta della soglia del primo modello su un insieme di validazione.

```text
Dataset di eventi (607.656 righe)
                |
Interpretazione degli eventi e ricostruzione delle partite (380)
                |
Statistiche per squadra e partita (760 righe)
                |
Analisi esplorativa e visualizzazioni
                |
XGBoost a due stadi: pareggio/non pareggio → vittoria/sconfitta
                |
Valutazione originale su un test set di righe squadra–partita
```

## Risultati riportati nel notebook

| Metrica della valutazione originale | Valore |
| --- | ---: |
| Accuracy finale sul test (modello a due stadi) | **78,29%** |
| Balanced accuracy finale sul test | **78,36%** |
| F1 macro finale sul test | **0,7920** |
| Osservazioni nel test finale | **152** |
| Balanced accuracy in validazione, stadio 1 (pareggio/non pareggio) | **94,52%** |
| Accuracy in validazione, stadio 2 (vittoria/sconfitta) | **81,36%** |

**Interpretazione:** i valori provengono dagli output salvati nel notebook e non sono stati ricalcolati durante la preparazione del README. Il test set contiene osservazioni squadra–partita, che non corrispondono necessariamente a partite tutte distinte.

## Limiti metodologici

- **Dipendenza tra osservazioni della stessa partita:** ogni partita produce due righe, una per squadra, mentre la suddivisione originale tra training, validazione e test è effettuata **per riga**. Le due osservazioni della stessa partita possono quindi finire in insiemi differenti. Per una valutazione più rigorosa sarebbe opportuno suddividere i dati per `match_id`, usando una strategia che tenga unite le osservazioni di ciascuna partita.
- **Statistiche raccolte a partita conclusa:** molte variabili sono aggregate a partire dagli eventi dell'**intera partita** (per esempio tiri e passaggi). Il compito svolto è quindi più correttamente una **classificazione delle prestazioni osservate** che una previsione prima del calcio d'inizio. Per prevedere davvero un risultato futuro servirebbero dati disponibili prima dell'incontro e una valutazione rispettosa dell'ordine temporale.
- **Riproducibilità:** il file originale dei dati **non è incluso** nel repository. Per eseguire il progetto è necessario procurarsi il CSV di partenza. I risultati possono inoltre dipendere dalle versioni delle librerie utilizzate.

Questi limiti non riducono il valore delle attività di preparazione dei dati, analisi e modellazione, ma sono importanti per interpretare correttamente le prestazioni del classificatore.

## Struttura del repository

| File | Contenuto |
| --- | --- |
| [`premier_league.ipynb`](premier_league.ipynb) | Notebook **finale presentato all’esame**, con codice Python, commenti, analisi esplorative, visualizzazioni e output della valutazione dei modelli. |
| `README.md` | Presentazione sintetica del progetto e note sulla metodologia. |

Il dataset originale `20-21_plEventsData.csv` **non è incluso**, perché supera i limiti pratici di caricamento adottati per il repository. Il notebook rimane consultabile su GitHub anche senza il CSV; per **rieseguire** tutte le celle occorre invece procurarsi i dati originali.

## Come consultare ed eseguire il notebook

1. Aprire [`premier_league.ipynb`](premier_league.ipynb) direttamente su GitHub per leggere le analisi e gli output salvati.
2. Per rieseguirlo, scaricare il notebook e aprirlo con Jupyter Notebook, JupyterLab oppure Visual Studio Code con il supporto Jupyter.
3. Collocare `20-21_plEventsData.csv` nella **stessa cartella** del notebook. Il file è richiamato con `pd.read_csv("20-21_plEventsData.csv", low_memory=False)`.
4. Predisporre un ambiente Python con le librerie importate nel notebook ed eseguire le celle in ordine. Le procedure di ottimizzazione degli iperparametri possono richiedere tempo.

Il repository documenta un progetto universitario: **non è un pacchetto Python installabile né un'applicazione pronta all'uso**. La fonte e i diritti di ridistribuzione del dataset non sono documentati nei materiali disponibili.

## Competenze messe in pratica

Pulizia di dati a livello di evento · Modellazione e aggregazione dei dati · Feature engineering · Analisi calcistica · Visualizzazione dei dati · Classificazione supervisionata · Valutazione dei modelli · Analisi critica della metodologia di validazione.

## Contesto e contributi

Progetto universitario realizzato in **un gruppo di tre persone**. Il repository contiene i materiali prodotti dal gruppo. Eventuali descrizioni dei contributi individuali devono rispecchiare le attività effettivamente svolte. La pubblicazione del codice non implica il possesso di una licenza per ridistribuire il dataset originale.
