use rand::seq::SliceRandom;
use rand::Rng;

use crate::engine::Value;

pub struct Neuron {
    pub weights: Vec<Value>,
    pub bias: Value,
    pub activation: String,
}

impl Neuron {
    pub fn new(num_in: usize, activation: &str) -> Self {
        let mut rng = rand::thread_rng();
        let weights = (0..num_in)
            .map(|i| Value::with_label(rng.gen_range(-1.0..1.0), &format!("w{}", i)))
            .collect();
        Self {
            weights,
            bias: Value::with_label(0.0, "b"),
            activation: activation.to_string(),
        }
    }

    pub fn forward(&self, x: &[Value]) -> Value {
        let mut raw = self.bias.clone();
        for (weight, input) in self.weights.iter().zip(x.iter()) {
            raw = raw + weight.clone() * input.clone();
        }

        match self.activation.as_str() {
            "sigmoid" => raw.sigmoid(),
            "relu" => raw.relu(),
            _ => raw.tanh(),
        }
    }

    pub fn parameters(&self) -> Vec<Value> {
        let mut params = self.weights.clone();
        params.push(self.bias.clone());
        params
    }
}

impl std::fmt::Display for Neuron {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "\nneuron(weights: {}) - activation: '{}'",
            self.weights.len(),
            capitalize(&self.activation)
        )
    }
}

pub struct Layer {
    pub neurons: Vec<Neuron>,
}

impl Layer {
    pub fn new(num_in: usize, num_out: usize, activation: &str) -> Self {
        Self {
            neurons: (0..num_out)
                .map(|_| Neuron::new(num_in, activation))
                .collect(),
        }
    }

    pub fn forward(&self, x: &[Value]) -> Vec<Value> {
        self.neurons.iter().map(|neuron| neuron.forward(x)).collect()
    }

    pub fn parameters(&self) -> Vec<Value> {
        let mut params = Vec::new();
        for neuron in &self.neurons {
            params.extend(neuron.parameters());
        }
        params
    }
}

impl std::fmt::Display for Layer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let neurons: Vec<String> = self.neurons.iter().map(|n| n.to_string()).collect();
        write!(
            f,
            "\nlayer of {} neurons: [{}]",
            self.neurons.len(),
            neurons.join(", ")
        )
    }
}

pub struct Network {
    pub layers: Vec<Layer>,
    pub unit_activation: String,
}

impl Network {
    pub fn new(num_in: usize, layer_sizes: &[usize], unit_activation: &str) -> Self {
        let mut size = vec![num_in];
        size.extend_from_slice(layer_sizes);

        let layers = (0..layer_sizes.len())
            .map(|i| Layer::new(size[i], size[i + 1], unit_activation))
            .collect();

        Self {
            layers,
            unit_activation: unit_activation.to_string(),
        }
    }

    pub fn forward(&self, x: &[Value]) -> Vec<Value> {
        let mut current = x.to_vec();
        for layer in &self.layers {
            current = layer.forward(&current);
        }
        current
    }

    pub fn parameters(&self) -> Vec<Value> {
        let mut params = Vec::new();
        for layer in &self.layers {
            params.extend(layer.parameters());
        }
        params
    }

    pub fn zero_grad(&self) {
        for param in self.parameters() {
            param.zero_gradient();
        }
    }

    fn score(&self, row: &[f64]) -> Value {
        let inputs: Vec<Value> = row.iter().map(|v| Value::new(*v)).collect();
        self.forward(&inputs)
            .into_iter()
            .next()
            .expect("network must have at least one output")
    }

    pub fn summary(&self) {
        println!("Network architecture:");
        println!("--------------------------------------");
        println!("{}", self);
        println!("--------------------------------------");
        println!("Total number of parameters: {}", self.parameters().len());
        println!("Total number of dense layers: {}", self.layers.len());
        println!(
            "Total number of neurons: {}",
            self.layers.iter().map(|layer| layer.neurons.len()).sum::<usize>()
        );
        println!(
            "Total number of weights: {}",
            self.layers
                .iter()
                .map(|layer| layer.neurons[0].weights.len())
                .sum::<usize>()
        );
        println!(
            "Total number of biases: {}",
            self.layers.iter().map(|layer| layer.neurons.len()).sum::<usize>()
        );
    }

    pub fn mse_loss(&self, x: &[Vec<f64>], y: &[f64]) -> Value {
        let mut loss = Value::new(0.0);
        for (row, target) in x.iter().zip(y.iter()) {
            let score = self.score(row);
            let diff = score.clone() - *target;
            loss = loss + diff.clone() * diff;
        }

        let lambda = 0.00001;
        let mut reg_loss = Value::new(0.0);
        for param in self.parameters() {
            reg_loss = reg_loss + param.clone() * param.clone();
        }

        (loss + reg_loss * lambda) / (2.0 * x.len() as f64)
    }

    pub fn svm_loss(
        &self,
        x: &[Vec<f64>],
        y: &[f64],
        batch_size: Option<usize>,
    ) -> (Value, f64) {
        let indices: Vec<usize> = match batch_size {
            Some(batch) => {
                let mut indices: Vec<usize> = (0..x.len()).collect();
                indices.shuffle(&mut rand::thread_rng());
                indices.truncate(batch.min(x.len()));
                indices
            }
            None => (0..x.len()).collect(),
        };

        let mut losses = Vec::new();
        let mut correct = 0usize;
        for &index in &indices {
            let score = self.score(&x[index]);
            let margin = Value::new(1.0).add(&(score.clone() * (-y[index])));
            losses.push(margin.relu());
            if (y[index] > 0.0) == (score.data() > 0.0) {
                correct += 1;
            }
        }

        let count = losses.len().max(1);
        let mut data_loss = Value::new(0.0);
        for loss in &losses {
            data_loss = data_loss + loss.clone();
        }
        data_loss = data_loss * (1.0 / count as f64);

        let lambda = 0.0001;
        let mut reg_loss = Value::new(0.0);
        for param in self.parameters() {
            reg_loss = reg_loss + param.clone() * param.clone();
        }

        let total_loss = data_loss + reg_loss * lambda;
        (total_loss, correct as f64 / count as f64)
    }

    pub fn fit(&self, x: &[Vec<f64>], y: &[f64], epochs: usize) {
        for epoch in 0..epochs {
            let loss = self.mse_loss(x, y);

            self.zero_grad();
            loss.backward();

            let alpha = 1.0 - 0.9 * epoch as f64 / 100.0;
            for param in self.parameters() {
                param.set_data(param.data() - alpha * param.gradient());
            }

            if epoch % 10 == 0 {
                println!("Epoch: {}/{} - Loss: {:.4}", epoch + 1, epochs, loss.data());
            }
        }
    }

    pub fn predict(&self, x: &[Vec<f64>]) -> Vec<f64> {
        x.iter().map(|row| self.score(row).data()).collect()
    }
}

impl std::fmt::Display for Network {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let layers: Vec<String> = self.layers.iter().map(|layer| layer.to_string()).collect();
        write!(
            f,
            "neural network of {} layers: [{}]",
            self.layers.len(),
            layers.join(", ")
        )
    }
}

fn capitalize(text: &str) -> String {
    let mut chars = text.chars();
    match chars.next() {
        Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
        None => String::new(),
    }
}
