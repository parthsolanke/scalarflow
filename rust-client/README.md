# ScalarFlow Rust client

A small command-line client for the ScalarFlow HTTP server, written in Rust. It
talks to the JSON endpoints exposed by `python -m scalarflow.server`.

## Build

```bash
cd rust-client
cargo build --release
```

The binary is written to `target/release/scalarflow-client`.

## Usage

Start the server first:

```bash
python -m scalarflow.server --host 127.0.0.1 --port 8000
```

Then run the client (the base URL defaults to `http://127.0.0.1:8000`, and can
be overridden with `--url` or the `SCALARFLOW_URL` environment variable):

```bash
# Server status and current model configuration
cargo run -- health

# Replace the in-memory network
cargo run -- model --num-in 2 --layer-sizes 4,2,1 --activation tanh

# Train on an XOR-style dataset
cargo run -- train \
  --inputs '[[0,0],[0,1],[1,0],[1,1]]' \
  --outputs '[0,1,1,0]' \
  --epochs 100

# Run a forward pass
cargo run -- predict --inputs '[[0,0],[1,0]]'
```

Every command prints the server's JSON response, pretty-printed. Add
`--url https://example.com` (or set `SCALARFLOW_URL`) to target a different
server.

## Commands

| Command | Endpoint | Notes |
| --- | --- | --- |
| `health` | `GET /health` | server status and model info |
| `model` | `POST /model` | all options optional; omitted fields keep their current values |
| `train` | `POST /train` | requires `--inputs` and `--outputs`; optional `--epochs` |
| `predict` | `POST /predict` | requires `--inputs` |

`--inputs` and `--outputs` take JSON literals, matching the request bodies in
the main README.
