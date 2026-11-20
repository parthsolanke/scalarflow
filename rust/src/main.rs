use scalarflow::engine::Value;
use scalarflow::nn::Network;

fn main() {
    let a = Value::new(2.0);
    let b = Value::new(3.0);
    let c = a.clone() + b.clone();
    let d = a * b;
    d.backward();
    println!("c = {}", c);
    println!("d = {}", d);

    let model = Network::new(2, &[4, 2, 1], "relu");
    model.summary();

    let inputs = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];
    let outputs = vec![0.0, 1.0, 1.0, 0.0];

    model.fit(&inputs, &outputs, 100);
    println!("predictions: {:?}", model.predict(&inputs));
}
