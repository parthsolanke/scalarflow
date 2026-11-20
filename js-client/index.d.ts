export type Scalar = number | value;
export type Activation = '' | 'sigmoid' | 'tanh' | 'relu' | string;

export class value {
  constructor(
    data: number,
    _parents?: Iterable<value>,
    _operation?: string,
    label?: string
  );

  data: number;
  _prev: Set<value>;
  _operation: string;
  label: string;
  gradient: number;
  _backward: () => void;
  readonly _id: number;

  toString(): string;

  add(other: Scalar): value;
  radd(other: Scalar): value;
  neg(): value;
  sub(other: Scalar): value;
  rsub(other: Scalar): value;
  mul(other: Scalar): value;
  rmul(other: Scalar): value;
  pow(other: number): value;
  truediv(other: Scalar): value;
  div(other: Scalar): value;
  rtruediv(other: Scalar): value;
  rdiv(other: Scalar): value;
  exp(): value;

  tanh(): value;
  relu(): value;
  sigmoid(): value;

  backward(): void;

  _trace(): { nodes: Set<value>; edges: Array<[value, value]> };
  toDot(): string;
  draw_graph(): string;
}

export { value as Value };

export class module {
  zero_grad(): void;
  parameters(): value[];
  forward(x: Scalar[]): value | value[];
  call(x: Scalar[]): value | value[];
}

export class neuron extends module {
  constructor(num_in: number, _activation?: Activation);

  weights: value[];
  bias: value;
  _activation: Activation;

  forward(x: Scalar[]): value;
  call(x: Scalar[]): value;
  parameters(): value[];
  toString(): string;
}

export class layer extends module {
  constructor(num_in: number, num_out: number, _activation?: Activation);

  neurons: neuron[];

  forward(x: Scalar[]): value | value[];
  call(x: Scalar[]): value | value[];
  parameters(): value[];
  toString(): string;
}

export class network extends module {
  constructor(num_in: number, layer_sizes: number[], unit_activation?: Activation);

  unit_activation: Activation;
  layers: layer[];

  forward(x: Scalar[]): value | value[];
  call(x: Scalar[]): value | value[];
  parameters(): value[];
  toString(): string;

  summary(): void;
  svm_loss(
    x: Scalar[][],
    y: Array<Scalar | Scalar[]>,
    batch_size?: number | null
  ): [value, number];
  mse_loss(x: Scalar[][], y: Array<Scalar | Scalar[]>): value;
  fit(x: Scalar[][], y: Array<Scalar | Scalar[]>, epochs: number): void;
  predict(x: Scalar[][]): number[];
}

export declare const engine: {
  value: typeof value;
};

export declare const nn: {
  module: typeof module;
  neuron: typeof neuron;
  layer: typeof layer;
  network: typeof network;
};

declare const scalarflow: {
  engine: typeof engine;
  nn: typeof nn;
  value: typeof value;
  Value: typeof value;
  module: typeof module;
  neuron: typeof neuron;
  layer: typeof layer;
  network: typeof network;
};

export default scalarflow;
