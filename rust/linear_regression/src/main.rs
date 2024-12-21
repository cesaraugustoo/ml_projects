fn main() {
    // Dataset representation
    let x: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0]; // Variable declarations using `let`, these are not mutable. If so, one would declare them as `let mut`
    let y: Vec<f64> = vec![2.0, 4.0, 6.0, 8.0]; // `Vec` is a growable array type, here storing floating-point numbers (f64), like Python's `list`. The macro `vec!` creates a new vector initialized with the provided values

    // Parameter initialization
    let mut m = 0.0; // `mut` allows the variable to have its value changed later (mutable variable)
    let mut b = 0.0; // Floating-point literals default to f64

    let learning_rate = 0.01;
}

fn lin_reg(x: &Vec<f64>, m: f64, b: f64) -> Vec<f64> {
    // `&Vec<f64>` means a reference is being passed to the vector, avoiding data copying. This means the function borrows the vector; it doesn't take ownership of it. This is important for performance because it avoids unnecessary copying of the vector.
    // Since `m` and `b` are immutable parameters, they are passed by value.
    // `-> Vec<f64>` indicates that the function returns a new vector of 64-bit floating-point numbers.
    x.iter() // `x`, refers to the input vector. 
             // `.iter()` method creates an iterator over the elements of the vector `x`. 
             // An iterator is a sequence of values that can be processed one at a time. This is a zero-cost abstraction in Rust: it doesn't incur runtime overhead compared to a manual loop. Crucially, `iter()` yields references to the elements of `x` (i.e., &f64).
        .map(|&xi| m * xi + b) // This is a method adapter that transforms the iterator.
                               // `.map(...)`  method takes a closure (an anonymous function) as an argument. It applies this closure to each element produced by the iterator.
                               // `|&xi|` is the closure's parameter list:
                                    // `|...|`, delimits the closure's parameter list;
                                    // `&xi` is a pattern that deconstructs the reference yielded by `x.iter()`. Because `iter()` yields `&f64` (a reference to an f64), we use `&xi` to get the actual f64 value (by dereferencing the reference). If we just used `xi`, the type would be `&&f64` (a reference to a reference to an f64), which would require an extra dereference later. This is often a source of confusion for newcomers.
        .collect() // `.collect()` is another method adapter that consumes the iterator and collects its elements into a new collection.
                   // This method takes the iterator of f64 values produced by map and creates a new Vec<f64> containing those values.
}

fn mse()