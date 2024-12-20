use linfa_trees::DecisionTrees;
use ndarray::Array2;
use ndarray_csv::Array2Reader;
use std::fs::File;

fn load_data(file_path: &str) -> Result<Array2<f64>, Box<dyn std::error::Error>> {
    let file = File::open(file_path)?;
    let mut reader = csv::Reader::from_reader(file);
    let array = reader.deserialize_array2_dynamic()?;
    Ok(array)
}

fn build_model() -> DecisionTree {
    DecisionTree::params()
        .min_samples_leaf(1)
        .max_depth(Some(5))
        .fit(&train)
        .unwrap()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data: Array2<f64> = load_data("/home/cesarsouza/ml_projects/rust/rust-ml-101/data/data.csv")?;
    println!("{:?}", data);
    Ok(())
}
