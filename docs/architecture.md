# Architecture

```mermaid
flowchart TD
  U[User] --> A[Accounts]
  U --> V[Video Upload]
  V --> R[AnalysisRecord]
  R --> P[Preprocessing service]
  P --> F[ResNeXt50 feature extractor]
  F --> C[LSTM classifier]
  C --> R
  R --> H[Owner-only history]
```

`accounts` owns registration/login. `analysis` owns records, upload validation, owner checks, and views. `ml` is framework-independent and contains preprocessing, model construction, checkpoint loading, training, evaluation, and metrics. SQLite is the development database; media files are kept outside source control.

