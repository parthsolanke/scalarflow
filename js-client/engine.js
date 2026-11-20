let nextId = 0;

function asValue(other) {
  return other instanceof value ? other : new value(other);
}

function format(value) {
  return Number(value).toFixed(4);
}

class value {
  constructor(data, _parents = [], _operation = '', label = '') {
    this.data = data;
    this._prev = new Set(_parents);
    this._operation = _operation;
    this.label = label;
    this.gradient = 0;
    this._backward = () => {};
    this._id = nextId++;
  }

  toString() {
    return `value(data = ${this.data}, grad = ${this.gradient})`;
  }

  /* <overloading operators> */
  add(other) {
    other = asValue(other);
    const sum = new value(this.data + other.data, [this, other], '+');

    sum._backward = () => {
      this.gradient += sum.gradient;
      other.gradient += sum.gradient;
    };

    return sum;
  }

  radd(other) {
    return asValue(other).add(this);
  }

  neg() {
    return this.mul(-1);
  }

  sub(other) {
    return this.add(asValue(other).neg());
  }

  rsub(other) {
    return asValue(other).add(this.neg());
  }

  mul(other) {
    other = asValue(other);
    const product = new value(this.data * other.data, [this, other], '*');

    product._backward = () => {
      this.gradient += other.data * product.gradient;
      other.gradient += this.data * product.gradient;
    };

    return product;
  }

  rmul(other) {
    return asValue(other).mul(this);
  }

  pow(other) {
    if (typeof other !== 'number') {
      throw new Error('only supporting int and float powers');
    }
    const out = new value(this.data ** other, [this], `**${other}`);

    out._backward = () => {
      this.gradient += other * this.data ** (other - 1) * out.gradient;
    };

    return out;
  }

  truediv(other) {
    return this.mul(asValue(other).pow(-1));
  }

  div(other) {
    return this.truediv(other);
  }

  rtruediv(other) {
    return asValue(other).mul(this.pow(-1));
  }

  rdiv(other) {
    return this.rtruediv(other);
  }

  exp() {
    const x = this.data;
    const e = new value(Math.exp(x), [this], 'exp');

    e._backward = () => {
      this.gradient += e.data * e.gradient;
    };

    return e;
  }
  /* </overloading operators> */

  /* <activation functions> */
  tanh() {
    const x = this.data;
    const n = Math.tanh(x);
    const t = new value(n, [this], 'tanh');

    t._backward = () => {
      this.gradient += (1 - n ** 2) * t.gradient;
    };

    return t;
  }

  relu() {
    const out = new value(this.data < 0 ? 0 : this.data, [this], 'relu');

    out._backward = () => {
      this.gradient += out.data > 0 ? out.gradient : 0;
    };

    return out;
  }

  sigmoid() {
    const x = this.data;
    const s = new value(1 / (1 + Math.exp(-x)), [this], 'sigmoid');

    s._backward = () => {
      this.gradient += s.data * (1 - s.data) * s.gradient;
    };

    return s;
  }
  /* </activation functions> */

  /* <backpropagation> */
  backward() {
    const topo = [];
    const visited = new Set();

    const build = (v) => {
      if (!visited.has(v)) {
        visited.add(v);
        for (const child of v._prev) {
          build(child);
        }
        topo.push(v);
      }
    };
    build(this);

    this.gradient = 1;
    for (let i = topo.length - 1; i >= 0; i--) {
      topo[i]._backward();
    }
  }
  /* </backpropagation> */

  /* <visualization methods> */
  _trace() {
    const nodes = new Set();
    const edges = new Set();

    const build = (v) => {
      if (!nodes.has(v)) {
        nodes.add(v);
        for (const parent of v._prev) {
          edges.add([parent, v]);
          build(parent);
        }
      }
    };
    build(this);

    return { nodes, edges };
  }

  toDot() {
    const { nodes, edges } = this._trace();
    const lines = ['digraph G {', '  rankdir=LR;'];

    for (const n of nodes) {
      const uid = JSON.stringify(`v${n._id}`);
      lines.push(
        `  ${uid} [label="{ ${n.label} | data: ${format(n.data)} | grad: ${format(n.gradient)} }", shape=record, style=filled, fillcolor=lightblue];`
      );
      if (n._operation) {
        const opId = JSON.stringify(`v${n._id}_${n._operation}`);
        lines.push(
          `  ${opId} [label=${JSON.stringify(n._operation)}, shape=circle, style=filled, fillcolor=orange];`
        );
        lines.push(`  ${opId} -> ${uid} [style=dashed];`);
      }
    }

    for (const [n1, n2] of edges) {
      const from = JSON.stringify(`v${n1._id}`);
      const to = JSON.stringify(`v${n2._id}_${n2._operation}`);
      lines.push(`  ${from} -> ${to} [style=dashed];`);
    }

    lines.push('}');
    return lines.join('\n');
  }

  draw_graph() {
    return this.toDot();
  }
  /* </visualization methods> */
}

export { value };
