# Examples

Run these scripts from the repository root after installing the package.
The [user guide](../docs/guide/index.md) covers installation and configuration.
The [walkthrough](../docs/guide/walkthrough.md) contains additional complete
scripts for Python node queries and precise source amendments.

| Script | Purpose | Requirements |
| --- | --- | --- |
| [offline.py](offline.py) | Retrieve two seeds, apply an amendment, query a descendant, and synthesize with scripted models | Core installation, running Docker, and the trusted local Python image. No credentials or hosted calls |
| [recursive_memory.py](recursive_memory.py) | Ingest sources and answer through `LLGM` | Docker, local Python image, configured hosted models, provider extra, and credentials |
| [modal_retrieval.py](modal_retrieval.py) | Query a persisted ColBERTv2/PLAID index and resolve canonical passage IDs | Modal extra, authenticated workspace, deployed functions, and a completed GPU test's local artifacts |

The default Docker image is `python:3.12-slim`. Prepare it locally before running.
The runtime never pulls an image automatically.

```sh
python examples/offline.py
```

For the hosted example, use the
[configuration guide](../docs/guide/configuration.md) to select providers and
model IDs. Export credentials and settings, then run:

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
