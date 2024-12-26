use linear_regression::LinearRegression;
use ndarray::{arr2, Array1, Array2};
use std::error::Error;

// Function to normalize features
fn normalize_features(x: &Array2<f64>) -> (Array2<f64>, Array1<f64>, Array1<f64>) {
    let mut means = Array1::zeros(x.ncols());
    let mut stds = Array1::zeros(x.ncols());
    
    // Calculate mean and std for each feature
    for j in 0..x.ncols() {
        let column = x.column(j);
        means[j] = column.mean().unwrap();
        stds[j] = column.iter()
            .map(|&x| (x - means[j]).powi(2))
            .sum::<f64>()
            .sqrt() / (column.len() as f64).sqrt();
    }
    
    // Create normalized features array
    let mut x_normalized = Array2::zeros(x.dim());
    for i in 0..x.nrows() {
        for j in 0..x.ncols() {
            x_normalized[[i, j]] = (x[[i, j]] - means[j]) / stds[j];
        }
    }
    
    (x_normalized, means, stds)
}

// Function to normalize new data using existing means and stds
fn normalize_new_data(x: &Array2<f64>, means: &Array1<f64>, stds: &Array1<f64>) -> Array2<f64> {
    let mut x_normalized = Array2::zeros(x.dim());
    for i in 0..x.nrows() {
        for j in 0..x.ncols() {
            x_normalized[[i, j]] = (x[[i, j]] - means[j]) / stds[j];
        }
    }
    x_normalized
}

fn main() -> Result<(), Box<dyn Error>> {
    // Sample housing data: [square_footage, bedrooms]
    let x_train = arr2(&[
        [1200.0, 2.0],
        [1500.0, 3.0],
        [2000.0, 3.0],
        [1700.0, 3.0],
        [1100.0, 2.0],
        [1600.0, 3.0],
        [2300.0, 4.0],
        [1900.0, 3.0],
        [2100.0, 4.0],
        [1250.0, 2.0],
    ]);

    // House prices in thousands of dollars
    let y_train = Array1::from(vec![
        200.0, 250.0, 320.0, 280.0, 190.0,
        260.0, 355.0, 310.0, 330.0, 205.0,
    ]);

    // Normalize features
    println!("Normalizing features...");
    let (x_train_norm, means, stds) = normalize_features(&x_train);
    
    // Create and train the model
    let mut model = LinearRegression::new(2, 0.01);
    
    println!("Training model...");
    let history = model.train(&x_train_norm, &y_train, 1000)?;
    
    // Print training results
    println!("Training completed!");
    println!("Initial loss: {:.2}", history[0]);
    println!("Final loss: {:.2}", history[history.len() - 1]);

    // Make predictions on some test cases
    let x_test = arr2(&[
        [1800.0, 3.0], // Medium house
        [2500.0, 4.0], // Large house
        [1000.0, 2.0], // Small house
    ]);

    // Normalize test data using training means and stds
    let x_test_norm = normalize_new_data(&x_test, &means, &stds);

    println!("\nMaking predictions...");
    let predictions = model.predict(&x_test_norm)?;

    println!("\nPredicted prices:");
    for (_i, (&pred, house)) in predictions.iter().zip(x_test.rows()).enumerate() {
        println!("{:.0} sqft, {} bed house: ${:.2}k", 
                house[0], 
                house[1], 
                pred);
    }

    // Calculate and print R-squared for training data
    let train_predictions = model.predict(&x_train_norm)?;
    let r_squared = model.r_squared(&train_predictions, &y_train);
    println!("\nModel R-squared: {:.4}", r_squared);

    // Print feature importance (normalized coefficients)
    println!("\nFeature importance (normalized coefficients):");
    println!("Square footage: {:.4}", model.weights[0]);
    println!("Bedrooms: {:.4}", model.weights[1]);

    Ok(())
}