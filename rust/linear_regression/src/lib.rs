use ndarray::{Array1, Array2};
use std::error::Error;

#[derive(Debug)]
pub struct LinearRegression {
    pub weights: Array1<f64>,
    pub bias: f64,
    learning_rate: f64,
}

#[derive(Debug)]
pub enum LinearRegressionError {
    DimensionMismatch {
        expected: usize,
        found: usize,
        context: &'static str,
    },
    EmptyData,
    NumericalError(&'static str),
}

impl std::fmt::Display for LinearRegressionError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::DimensionMismatch { expected, found, context } => {
                write!(f, "Dimension mismatch in {}: expected {}, found {}", 
                       context, expected, found)
            }
            Self::EmptyData => write!(f, "Empty data provided"),
            Self::NumericalError(msg) => write!(f, "Numerical error: {}", msg),
        }
    }
}

impl Error for LinearRegressionError {}

impl LinearRegression {
    pub fn new(n_features: usize, learning_rate: f64) -> Self {
        Self {
            weights: Array1::zeros(n_features),
            bias: 0.0,
            learning_rate,
        }
    }

    pub fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>, LinearRegressionError> {
        if x.ncols() != self.weights.len() {
            return Err(LinearRegressionError::DimensionMismatch {
                expected: self.weights.len(),
                found: x.ncols(),
                context: "number of features in prediction",
            });
        }
        
        Ok(x.dot(&self.weights) + self.bias)
    }

    pub fn mse_loss(&self, predictions: &Array1<f64>, y: &Array1<f64>) -> f64 {
        let errors = predictions - y;
        errors.mapv(|e| e * e).mean().unwrap_or(f64::INFINITY)
    }

    pub fn r_squared(&self, predictions: &Array1<f64>, y: &Array1<f64>) -> f64 {
        let y_mean = y.mean().unwrap_or(0.0);
        let ss_tot = y.iter()
            .map(|&y_i| (y_i - y_mean).powi(2))
            .sum::<f64>();
        let ss_res = predictions.iter()
            .zip(y.iter())
            .map(|(&pred, &actual)| (actual - pred).powi(2))
            .sum::<f64>();
        
        1.0 - (ss_res / ss_tot)
    }

    pub fn train(
        &mut self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        epochs: usize
    ) -> Result<Vec<f64>, LinearRegressionError> {
        // Validate input dimensions
        if x.nrows() != y.len() {
            return Err(LinearRegressionError::DimensionMismatch {
                expected: x.nrows(),
                found: y.len(),
                context: "number of samples in X and y",
            });
        }
        if x.ncols() != self.weights.len() {
            return Err(LinearRegressionError::DimensionMismatch {
                expected: self.weights.len(),
                found: x.ncols(),
                context: "number of features",
            });
        }
        if x.nrows() == 0 {
            return Err(LinearRegressionError::EmptyData);
        }

        let n_samples = x.nrows();
        let mut history = Vec::with_capacity(epochs);
        
        for _ in 0..epochs {
            let predictions = self.predict(x)?;
            let errors = &predictions - y;
            
            // Check for numerical stability
            if errors.iter().any(|&e| e.is_infinite() || e.is_nan()) {
                return Err(LinearRegressionError::NumericalError(
                    "Infinite or NaN values encountered during training"
                ));
            }

            let weight_gradients = x.t().dot(&errors) * (1.0 / n_samples as f64);
            let bias_gradient = errors.sum() * (1.0 / n_samples as f64);
            
            self.weights = &self.weights - &(weight_gradients * self.learning_rate);
            self.bias -= bias_gradient * self.learning_rate;
            
            let mse = self.mse_loss(&predictions, y);
            history.push(mse);
        }
        
        Ok(history)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;  // Added this import

    #[test]
    fn test_linear_regression() -> Result<(), Box<dyn Error>> {
        let x = arr2(&[
            [1.0, 2.0],
            [2.0, 4.0],
            [3.0, 6.0],
        ]);
        let y = Array1::from(vec![2.0, 4.0, 6.0]);

        let mut model = LinearRegression::new(2, 0.01);

        // Train model and handle potential errors
        let history = model.train(&x, &y, 100)?;
        
        // Test predictions
        let predictions = model.predict(&x)?;
        
        // Calculate R-squared
        let r2 = model.r_squared(&predictions, &y);
        assert!(r2 > 0.9); // Check for good fit

        // Check if error decreases
        assert!(history[0] > history[history.len() - 1]);
        
        Ok(())
    }

    #[test]
    fn test_dimension_mismatch() {
        let x = arr2(&[[1.0], [2.0]]); // 2x1 matrix
        let y = Array1::from(vec![2.0]); // 1 element
        
        let mut model = LinearRegression::new(1, 0.01);
        
        match model.train(&x, &y, 100) {
            Err(LinearRegressionError::DimensionMismatch { .. }) => (),
            _ => panic!("Expected dimension mismatch error"),
        }
    }
}