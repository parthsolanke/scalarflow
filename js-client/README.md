# scalarflow (JavaScript client)

An ESM JavaScript port of the ScalarFlow Python package. It exposes the same public
names and behavior as `scalarflow/engine.py` and `scalarflow/nn.py`.

## Install

```bash
npm install /path/to/this/repo/js-client
```

## Usage

### Autograd engine

Python overloads arithmetic operators; JavaScript callers use the matching methods
(`add`, `sub`, `mul`, `div`, `pow`, `neg`).

```js
import { value } from 'scalarflow';

const a = new value(2);
const b = new value(3);

const c = a.add(b);
const d = a.mul(b);

d.backward();

console.log(c.toString()); // value(data = 5, grad = 0)
console.log(d.toString()); // value(data = 6, grad = 1)
console.log(a.gradient);   // 3
```

The `value` class also provides `exp`, `tanh`, `relu`, `sigmoid`, `backward`,
`_trace`, `toDot` and `draw_graph`. `draw_graph` returns Graphviz DOT source as a
string, which you can render with Graphviz yourself.

### Neural networks

```js
import { network } from 'scalarflow';

const model = new network(2, [4, 2, 1], 'relu');

model.summary();

const inputs = [[0, 0], [0, 1], [1, 0], [1, 1]];
const outputs = [[0], [1], [1], [0]];

model.fit(inputs, outputs, 100);

console.log(model.predict(inputs));
```

`module`, `neuron` and `layer` are exported alongside `network`. `network` supports
`forward`/`call`, `parameters`, `zero_grad`, `summary`, `svm_loss`, `mse_loss`,
`fit` and `predict`. Losses return the same values as the Python package;
`svm_loss` returns a `[loss, accuracy]` tuple as an array.

The `engine` and `nn` namespaces are also exported, mirroring
`from scalarflow import engine, nn`:

```js
import { engine, nn } from 'scalarflow';

const x = new engine.value(1.5);
const model = new nn.network(1, [1]);
```
