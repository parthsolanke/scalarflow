import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from scalarflow.nn import network


DEFAULT_NUM_IN = 2
DEFAULT_LAYER_SIZES = [4, 2, 1]
DEFAULT_ACTIVATION = "tanh"
ACTIVATIONS = ("", "sigmoid", "tanh", "relu")
MAX_EPOCHS = 1_000_000


class ModelState:

    def __init__(self, num_in, layer_sizes, activation):
        self.lock = threading.Lock()
        self.model = None
        self.configure(num_in, layer_sizes, activation)

    def configure(self, num_in, layer_sizes, activation):
        self.num_in = num_in
        self.layer_sizes = list(layer_sizes)
        self.activation = activation
        self.model = network(num_in, self.layer_sizes, activation)

    def info(self):
        return {
            "num_in": self.num_in,
            "layer_sizes": self.layer_sizes,
            "activation": self.activation,
            "parameters": len(self.model.parameters()),
        }


def _is_number(item):
    return isinstance(item, (int, float)) and not isinstance(item, bool)


def _require_number_matrix(value, name):
    if not isinstance(value, list) or not value:
        raise ValueError(f"'{name}' must be a non-empty list of rows")
    for row in value:
        if not isinstance(row, list) or not all(_is_number(x) for x in row):
            raise ValueError(f"'{name}' rows must be lists of numbers")
    return value


def _require_number_list(value, name):
    if not isinstance(value, list) or not all(_is_number(x) for x in value):
        raise ValueError(f"'{name}' must be a list of numbers")
    return value


def _require_int(value, name, minimum=1, maximum=None):
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"'{name}' must be an integer >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"'{name}' must be <= {maximum}")
    return value


def make_handler(state):

    class Handler(BaseHTTPRequestHandler):
        server_version = "ScalarFlow/0.1"

        def log_message(self, format, *args):
            print(f"{self.address_string()} - {format % args}")

        def _send_json(self, status, payload):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self):
            length = int(self.headers.get("Content-Length") or 0)
            if length == 0:
                return {}
            raw = self.rfile.read(length)
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON body: {exc.msg}")
            if not isinstance(payload, dict):
                raise ValueError("request body must be a JSON object")
            return payload

        def do_GET(self):
            if self.path == "/health":
                self._send_json(200, {"status": "ok", "model": state.info()})
                return
            self._send_json(404, {"error": "not_found", "path": self.path})

        def do_POST(self):
            try:
                payload = self._read_json()
                if self.path == "/predict":
                    self._handle_predict(payload)
                elif self.path == "/train":
                    self._handle_train(payload)
                elif self.path == "/model":
                    self._handle_model(payload)
                else:
                    self._send_json(404, {"error": "not_found", "path": self.path})
            except ValueError as exc:
                self._send_json(400, {"error": "bad_request", "message": str(exc)})
            except Exception as exc:  # keep the server alive on unexpected errors
                self._send_json(500, {"error": "internal_error", "message": str(exc)})

        def _handle_predict(self, payload):
            inputs = _require_number_matrix(payload.get("inputs"), "inputs")
            for row in inputs:
                if len(row) != state.num_in:
                    raise ValueError(
                        f"each 'inputs' row must have {state.num_in} values"
                    )
            with state.lock:
                predictions = state.model.predict(inputs)
            self._send_json(200, {"predictions": predictions})

        def _handle_train(self, payload):
            inputs = _require_number_matrix(payload.get("inputs"), "inputs")
            outputs = _require_number_list(payload.get("outputs"), "outputs")
            epochs = _require_int(payload.get("epochs", 100), "epochs", maximum=MAX_EPOCHS)
            if len(inputs) != len(outputs):
                raise ValueError("'inputs' and 'outputs' must have the same length")
            for row in inputs:
                if len(row) != state.num_in:
                    raise ValueError(
                        f"each 'inputs' row must have {state.num_in} values"
                    )
            if state.layer_sizes[-1] != 1:
                raise ValueError(
                    "training requires the final layer size to be 1"
                )
            with state.lock:
                state.model.fit(inputs, outputs, epochs)
                loss = state.model.mse_loss(inputs, outputs).data
                predictions = state.model.predict(inputs)
            self._send_json(200, {
                "epochs": epochs,
                "loss": loss,
                "predictions": predictions,
                "model": state.info(),
            })

        def _handle_model(self, payload):
            num_in = _require_int(payload.get("num_in", state.num_in), "num_in")
            layer_sizes = payload.get("layer_sizes", state.layer_sizes)
            if not isinstance(layer_sizes, list) or not layer_sizes:
                raise ValueError("'layer_sizes' must be a non-empty list of integers")
            layer_sizes = [_require_int(size, "layer_sizes") for size in layer_sizes]
            activation = payload.get("activation", state.activation)
            if activation not in ACTIVATIONS:
                raise ValueError(
                    "'activation' must be one of " + ", ".join(repr(a) for a in ACTIVATIONS)
                )
            if layer_sizes[-1] != 1:
                raise ValueError("training requires the final layer size to be 1")
            with state.lock:
                state.configure(num_in, layer_sizes, activation)
                info = state.info()
            self._send_json(200, {"model": info})

    return Handler


def create_server(host="127.0.0.1", port=8000, num_in=DEFAULT_NUM_IN,
                  layer_sizes=None, activation=DEFAULT_ACTIVATION):
    if activation not in ACTIVATIONS:
        raise ValueError("unsupported activation: %r" % activation)
    state = ModelState(
        num_in,
        layer_sizes or DEFAULT_LAYER_SIZES,
        activation,
    )
    handler = make_handler(state)
    return ThreadingHTTPServer((host, port), handler)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Serve a ScalarFlow network over HTTP."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--num-in", type=int, default=DEFAULT_NUM_IN)
    parser.add_argument(
        "--layer-sizes",
        type=int,
        nargs="+",
        default=DEFAULT_LAYER_SIZES,
    )
    parser.add_argument("--activation", choices=ACTIVATIONS, default=DEFAULT_ACTIVATION)
    args = parser.parse_args(argv)

    httpd = create_server(
        host=args.host,
        port=args.port,
        num_in=args.num_in,
        layer_sizes=args.layer_sizes,
        activation=args.activation,
    )
    print(f"ScalarFlow server listening on http://{args.host}:{args.port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
