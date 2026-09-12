# Examples

Start with `recursive_memory.py` to store two related notes and ask a question
using your configured models. Run it from the repository root after following
the [quickstart](../docs/guide/quickstart.md). The first PyPI release is in
progress, so these examples currently use the checkout installation.

To explore evidence storage and corrections without model credentials or Docker,
use the [walkthrough](../docs/guide/walkthrough.md). The scripts below cover
answer generation and optional retrieval integrations.

| Script | Purpose | Requirements |
| --- | --- | --- |
| [offline.py](offline.py) | Retrieve two seeds, apply an amendment, query a descendant, and synthesize with scripted models | Core installation, running Docker, and the trusted local Python image. No credentials or hosted calls |
| [recursive_memory.py](recursive_memory.py) | Store related notes, ask a question, and inspect the answer's references | Docker, local Python image, configured hosted models, provider extra, and credentials |
| [modal_retrieval.py](modal_retrieval.py) | Query a persisted ColBERTv2/PLAID index and resolve canonical passage IDs | Modal extra, authenticated workspace, deployed functions, and a completed GPU test's local artifacts |

The default Docker image is `python:3.12-slim`. Prepare it locally before running.
The runtime never pulls an image automatically.

```sh
python examples/offline.py
```

For the hosted example, use the
[configuration guide](../docs/guide/configuration.md) to select providers and
model IDs. `LLGM.from_settings()` reads them when the application opens. Export
credentials and settings, then run:

```sh
python examples/recursive_memory.py
```

To use a local file, copy `.env.example` to `.env`, fill in the provider keys and
model IDs, and select it explicitly:

```sh
python examples/recursive_memory.py --env-file .env
```

Existing environment variables take precedence over file values. The example
does not discover or load `.env` without the option.

Hosted examples incur provider usage and persist evidence in the configured
workspace. Scripted output verifies local contracts, without measuring model
answer quality.
