use std::cell::RefCell;
use std::collections::HashSet;
use std::hash::{Hash, Hasher};
use std::rc::Rc;

type Backward = Rc<dyn Fn()>;

pub struct Node {
    pub data: f64,
    pub gradient: f64,
    pub prev: Vec<Value>,
    pub operation: String,
    pub label: String,
    backward: Option<Backward>,
}

#[derive(Clone)]
pub struct Value(Rc<RefCell<Node>>);

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for Value {}

impl Hash for Value {
    fn hash<H: Hasher>(&self, state: &mut H) {
        Rc::as_ptr(&self.0).hash(state);
    }
}

impl Value {
    pub fn new(data: f64) -> Self {
        Self(Rc::new(RefCell::new(Node {
            data,
            gradient: 0.0,
            prev: Vec::new(),
            operation: String::new(),
            label: String::new(),
            backward: None,
        })))
    }

    pub fn with_label(data: f64, label: &str) -> Self {
        let value = Self::new(data);
        value.0.borrow_mut().label = label.to_string();
        value
    }

    fn from_operation(data: f64, prev: Vec<Value>, operation: &str) -> Self {
        Self(Rc::new(RefCell::new(Node {
            data,
            gradient: 0.0,
            prev,
            operation: operation.to_string(),
            label: String::new(),
            backward: None,
        })))
    }

    pub fn data(&self) -> f64 {
        self.0.borrow().data
    }

    pub fn gradient(&self) -> f64 {
        self.0.borrow().gradient
    }

    pub fn set_data(&self, data: f64) {
        self.0.borrow_mut().data = data;
    }

    pub fn set_gradient(&self, gradient: f64) {
        self.0.borrow_mut().gradient = gradient;
    }

    pub fn zero_gradient(&self) {
        self.0.borrow_mut().gradient = 0.0;
    }

    pub fn add_gradient(&self, amount: f64) {
        self.0.borrow_mut().gradient += amount;
    }

    pub fn label(&self) -> String {
        self.0.borrow().label.clone()
    }

    pub fn operation(&self) -> String {
        self.0.borrow().operation.clone()
    }

    pub fn prev(&self) -> Vec<Value> {
        self.0.borrow().prev.clone()
    }

    pub fn add(&self, other: &Value) -> Value {
        let out = Value::from_operation(
            self.data() + other.data(),
            vec![self.clone(), other.clone()],
            "+",
        );
        let a = self.clone();
        let b = other.clone();
        let o = out.clone();
        out.0.borrow_mut().backward = Some(Rc::new(move || {
            let gradient = o.gradient();
            a.add_gradient(gradient);
            b.add_gradient(gradient);
        }));
        out
    }

    pub fn mul(&self, other: &Value) -> Value {
        let out = Value::from_operation(
            self.data() * other.data(),
            vec![self.clone(), other.clone()],
            "*",
        );
        let a = self.clone();
        let b = other.clone();
        let o = out.clone();
        out.0.borrow_mut().backward = Some(Rc::new(move || {
            let gradient = o.gradient();
            a.add_gradient(b.data() * gradient);
            b.add_gradient(a.data() * gradient);
        }));
        out
    }

    pub fn pow(&self, exponent: f64) -> Value {
        let out = Value::from_operation(
            self.data().powf(exponent),
            vec![self.clone()],
            &format!("**{}", exponent),
        );
        let base = self.clone();
        let o = out.clone();
        out.0.borrow_mut().backward = Some(Rc::new(move || {
            let gradient = o.gradient();
            base.add_gradient(exponent * base.data().powf(exponent - 1.0) * gradient);
        }));
        out
    }

    pub fn neg(&self) -> Value {
        self.mul(&Value::new(-1.0))
    }

    pub fn sub(&self, other: &Value) -> Value {
        self.add(&other.neg())
    }

    pub fn div(&self, other: &Value) -> Value {
        self.mul(&other.pow(-1.0))
    }

    pub fn exp(&self) -> Value {
        let out = Value::from_operation(self.data().exp(), vec![self.clone()], "exp");
        let base = self.clone();
        let o = out.clone();
        out.0.borrow_mut().backward = Some(Rc::new(move || {
            let gradient = o.gradient();
            base.add_gradient(o.data() * gradient);
        }));
        out
    }

    pub fn tanh(&self) -> Value {
        let e2 = (2.0 * self.data()).exp();
        let n = (e2 - 1.0) / (e2 + 1.0);
        let out = Value::from_operation(n, vec![self.clone()], "tanh");
        let base = self.clone();
        let o = out.clone();
        out.0.borrow_mut().backward = Some(Rc::new(move || {
            let gradient = o.gradient();
            base.add_gradient((1.0 - n * n) * gradient);
        }));
        out
    }

    pub fn relu(&self) -> Value {
        let data = if self.data() < 0.0 { 0.0 } else { self.data() };
        let out = Value::from_operation(data, vec![self.clone()], "relu");
        let base = self.clone();
        let o = out.clone();
        out.0.borrow_mut().backward = Some(Rc::new(move || {
            let gradient = o.gradient();
            let local = if o.data() > 0.0 { 1.0 } else { 0.0 };
            base.add_gradient(local * gradient);
        }));
        out
    }

    pub fn sigmoid(&self) -> Value {
        let s = 1.0 / (1.0 + (-self.data()).exp());
        let out = Value::from_operation(s, vec![self.clone()], "sigmoid");
        let base = self.clone();
        let o = out.clone();
        out.0.borrow_mut().backward = Some(Rc::new(move || {
            let gradient = o.gradient();
            base.add_gradient((s * (1.0 - s)) * gradient);
        }));
        out
    }

    pub fn backward(&self) {
        let mut topo = Vec::new();
        let mut visited = HashSet::new();
        Self::build_topo(self, &mut visited, &mut topo);

        self.set_gradient(1.0);
        for node in topo.iter().rev() {
            let backward = node.0.borrow().backward.clone();
            if let Some(backward) = backward {
                backward();
            }
        }
    }

    fn build_topo(value: &Value, visited: &mut HashSet<Value>, topo: &mut Vec<Value>) {
        if visited.insert(value.clone()) {
            for parent in value.prev() {
                Self::build_topo(&parent, visited, topo);
            }
            topo.push(value.clone());
        }
    }

    fn trace(
        value: &Value,
        nodes: &mut HashSet<Value>,
        edges: &mut HashSet<(Value, Value)>,
    ) {
        if nodes.insert(value.clone()) {
            for parent in value.prev() {
                edges.insert((parent.clone(), value.clone()));
                Self::trace(&parent, nodes, edges);
            }
        }
    }

    pub fn draw_graph(&self) -> String {
        let mut nodes = HashSet::new();
        let mut edges = HashSet::new();
        Self::trace(self, &mut nodes, &mut edges);

        let mut dot = String::from("digraph G {\n  rankdir=LR;\n");

        for node in &nodes {
            let uid = format!("{:p}", Rc::as_ptr(&node.0));
            dot.push_str(&format!(
                "  \"{}\" [label=\"{{ {} | data: {:.4} | grad: {:.4} }}\", shape=record, style=filled, fillcolor=lightblue];\n",
                uid,
                node.label(),
                node.data(),
                node.gradient()
            ));
            if !node.operation().is_empty() {
                let operation = node.operation();
                dot.push_str(&format!(
                    "  \"{}{}\" [label=\"{}\", shape=circle, style=filled, fillcolor=orange];\n",
                    uid, operation, operation
                ));
                dot.push_str(&format!(
                    "  \"{}{}\" -> \"{}\" [style=dashed];\n",
                    uid, operation, uid
                ));
            }
        }

        for (parent, child) in &edges {
            let parent_uid = format!("{:p}", Rc::as_ptr(&parent.0));
            let child_uid = format!("{:p}", Rc::as_ptr(&child.0));
            dot.push_str(&format!(
                "  \"{}\" -> \"{}{}\" [style=dashed];\n",
                parent_uid,
                child_uid,
                child.operation()
            ));
        }

        dot.push_str("}\n");
        dot
    }
}

impl std::fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "value(data = {}, grad = {})",
            self.data(),
            self.gradient()
        )
    }
}

impl std::ops::Add for Value {
    type Output = Value;

    fn add(self, rhs: Value) -> Value {
        Value::add(&self, &rhs)
    }
}

impl std::ops::Add<f64> for Value {
    type Output = Value;

    fn add(self, rhs: f64) -> Value {
        Value::add(&self, &Value::new(rhs))
    }
}

impl std::ops::Add<Value> for f64 {
    type Output = Value;

    fn add(self, rhs: Value) -> Value {
        Value::new(self).add(&rhs)
    }
}

impl std::ops::Sub for Value {
    type Output = Value;

    fn sub(self, rhs: Value) -> Value {
        Value::sub(&self, &rhs)
    }
}

impl std::ops::Sub<f64> for Value {
    type Output = Value;

    fn sub(self, rhs: f64) -> Value {
        Value::sub(&self, &Value::new(rhs))
    }
}

impl std::ops::Sub<Value> for f64 {
    type Output = Value;

    fn sub(self, rhs: Value) -> Value {
        Value::new(self).sub(&rhs)
    }
}

impl std::ops::Mul for Value {
    type Output = Value;

    fn mul(self, rhs: Value) -> Value {
        Value::mul(&self, &rhs)
    }
}

impl std::ops::Mul<f64> for Value {
    type Output = Value;

    fn mul(self, rhs: f64) -> Value {
        Value::mul(&self, &Value::new(rhs))
    }
}

impl std::ops::Mul<Value> for f64 {
    type Output = Value;

    fn mul(self, rhs: Value) -> Value {
        Value::new(self).mul(&rhs)
    }
}

impl std::ops::Div for Value {
    type Output = Value;

    fn div(self, rhs: Value) -> Value {
        Value::div(&self, &rhs)
    }
}

impl std::ops::Div<f64> for Value {
    type Output = Value;

    fn div(self, rhs: f64) -> Value {
        Value::div(&self, &Value::new(rhs))
    }
}

impl std::ops::Div<Value> for f64 {
    type Output = Value;

    fn div(self, rhs: Value) -> Value {
        Value::new(self).div(&rhs)
    }
}

impl std::ops::Neg for Value {
    type Output = Value;

    fn neg(self) -> Value {
        Value::neg(&self)
    }
}
