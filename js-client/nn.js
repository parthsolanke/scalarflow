import { value } from './engine.js';

function toNumber(v) {
  if (v instanceof value) return v.data;
  if (Array.isArray(v)) return toNumber(v[0]);
  return v;
}

function scoreOf(v) {
  return Array.isArray(v) ? v[0] : v;
}

export class module {
  zero_grad() {
    for (const p of this.parameters()) {
      p.gradient = 0;
    }
  }

  parameters() {
    return [];
  }

  forward(x) {
    throw new Error('forward is not implemented');
  }

  call(x) {
    return this.forward(x);
  }
}

export class neuron extends module {
  constructor(num_in, _activation = '') {
    super();
    this.weights = Array.from(
      { length: num_in },
      (_, i) => new value(Math.random() * 2 - 1, [], '', `w${i}`)
    );
    this.bias = new value(0, [], '', 'b');
    this._activation = _activation;
  }

  forward(x) {
    let raw = this.bias;
    const size = Math.min(this.weights.length, x.length);
    for (let i = 0; i < size; i++) {
      raw = raw.add(this.weights[i].mul(x[i]));
    }

    if (this._activation === 'sigmoid') return raw.sigmoid();
    if (this._activation === 'tanh') return raw.tanh();
    if (this._activation === 'relu') return raw.relu();
    return raw.tanh();
  }

  parameters() {
    return [...this.weights, this.bias];
  }

  toString() {
    const name =
      this._activation.charAt(0).toUpperCase() +
      this._activation.slice(1).toLowerCase();
    return `\nneuron(weights: ${this.weights.length}) - activation: '${name}'`;
  }
}

export class layer extends module {
  constructor(num_in, num_out, _activation = '') {
    super();
    this.neurons = Array.from(
      { length: num_out },
      () => new neuron(num_in, _activation)
    );
  }

  forward(x) {
    const out = this.neurons.map((n) => n.forward(x));
    return out.length === 1 ? out[0] : out;
  }

  parameters() {
    const params = [];
    for (const n of this.neurons) {
      params.push(...n.parameters());
    }
    return params;
  }

  toString() {
    return `\nlayer of ${this.neurons.length} neurons: [${this.neurons
      .map((n) => n.toString())
      .join(', ')}]`;
  }
}

export class network extends module {
  constructor(num_in, layer_sizes, unit_activation = '') {
    super();
    this.unit_activation = unit_activation;
    const size = [num_in, ...layer_sizes];
    this.layers = layer_sizes.map(
      (_, i) => new layer(size[i], size[i + 1], unit_activation)
    );
  }

  forward(x) {
    let out = x;
    for (const l of this.layers) {
      out = l.forward(out);
    }
    return out;
  }

  parameters() {
    const params = [];
    for (const l of this.layers) {
      params.push(...l.parameters());
    }
    return params;
  }

  toString() {
    return `neural network of ${this.layers.length} layers: [${this.layers
      .map((l) => l.toString())
      .join(', ')}\n]`;
  }

  summary() {
    console.log('Network architecture:');
    console.log('--------------------------------------');
    console.log(this.toString());
    console.log('--------------------------------------');
    console.log(`Total number of parameters: ${this.parameters().length}`);
    console.log(`Total number of dense layers: ${this.layers.length}`);
    console.log(
      `Total number of neurons: ${this.layers.reduce(
        (sum, l) => sum + l.neurons.length,
        0
      )}`
    );
    console.log(
      `Total number of weights: ${this.layers.reduce(
        (sum, l) => sum + l.neurons[0].weights.length,
        0
      )}`
    );
    console.log(
      `Total number of biases: ${this.layers.reduce(
        (sum, l) => sum + l.neurons.length,
        0
      )}`
    );
  }

  svm_loss(x, y, batch_size = null) {
    let Xb;
    let yb;

    if (batch_size == null) {
      Xb = x;
      yb = y;
    } else {
      const indices = Array.from({ length: x.length }, (_, i) => i);
      for (let i = indices.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [indices[i], indices[j]] = [indices[j], indices[i]];
      }
      const picked = indices.slice(0, batch_size);
      Xb = picked.map((i) => x[i]);
      yb = picked.map((i) => y[i]);
    }

    const inputs = Xb.map((xrow) =>
      xrow.map((v) => (v instanceof value ? v : new value(v)))
    );

    const scores = inputs.map((input) => this.forward(input));

    // svm "max-margin" loss
    const losses = scores.map((score, i) =>
      scoreOf(score).mul(-toNumber(yb[i])).add(1).relu()
    );
    let data_loss = new value(0);
    for (const loss of losses) {
      data_loss = data_loss.add(loss);
    }
    data_loss = data_loss.mul(1.0 / losses.length);

    // regularization
    const lam = 0.0001;
    let reg_loss = new value(0);
    for (const p of this.parameters()) {
      reg_loss = reg_loss.add(p.mul(p));
    }
    reg_loss = reg_loss.mul(lam);

    const total_loss = data_loss.add(reg_loss);

    let correct = 0;
    scores.forEach((score, i) => {
      if ((toNumber(yb[i]) > 0) === (scoreOf(score).data > 0)) correct++;
    });
    const accuracy = correct / scores.length;

    return [total_loss, accuracy];
  }

  mse_loss(x, y) {
    const scores = x.map((xi) => this.forward(xi));

    // mse
    let loss = new value(0);
    scores.forEach((score, i) => {
      const error = scoreOf(score).sub(toNumber(y[i]));
      loss = loss.add(error.mul(error));
    });

    // regularization
    const lam = 0.00001;
    let reg_loss = new value(0);
    for (const p of this.parameters()) {
      reg_loss = reg_loss.add(p.mul(p));
    }
    reg_loss = reg_loss.mul(lam);

    return loss.add(reg_loss).mul(1.0 / (2 * scores.length));
  }

  fit(x, y, epochs) {
    for (let epoch = 0; epoch < epochs; epoch++) {
      // forward pass
      const loss = this.mse_loss(x, y);

      // setting the gradient to zero for each parameter
      this.zero_grad();
      // backward pass
      loss.backward();

      // update parameters
      const alpha = 1.0 - (0.9 * epoch) / 100;
      for (const param of this.parameters()) {
        param.data -= alpha * param.gradient;
      }

      if (epoch % 10 === 0) {
        console.log(`Epoch: ${epoch + 1}/${epochs} - Loss: ${loss.data.toFixed(4)}`);
      }
    }
  }

  predict(x) {
    return x.map((xi) => scoreOf(this.forward(xi)).data);
  }
}
