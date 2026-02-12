use anyhow::{anyhow, Context, Result};
use csv::ReaderBuilder;
use log::{debug, info};
use polars::prelude::*;
use std::fs;
use std::io::Write;
use std::path::Path;
use xgboost::{Booster, DMatrix, parameters};

pub fn download_csv(url: &str, file_path: &str) -> Result<()> {
    info!("Starting download from {}", url);
    let response = reqwest::blocking::get(url)?;
    let content = response.bytes()?;

    // Create data folder if it doesn't exist
    if let Some(parent) = Path::new(file_path).parent() {
        fs::create_dir_all(parent)?;
        debug!("Created directory: {:?}", parent);
    }

    let mut file = fs::File::create(file_path)?;
    file.write_all(&content)?;

    info!("Downloaded CSV to {}", file_path);
    Ok(())
}

pub fn load_csv_to_dataframe(file_path: &str) -> Result<DataFrame> {
    info!("Loading CSV into Polars DataFrame from {}", file_path);
    let df = CsvReadOptions::default()
        .try_into_reader_with_file_path(Some(file_path.into()))?
        .finish()?;

    info!("Loaded DataFrame with shape: {:?}", df.shape());
    Ok(df)
}

pub fn print_csv_rows(file_path: &str, num_rows: usize) -> Result<()> {
    info!("Reading CSV from {}", file_path);
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .from_path(file_path)?;

    // Print header
    if let Ok(headers) = reader.headers() {
        println!("{}", headers.iter().collect::<Vec<_>>().join(","));
    }

    // Print specified number of rows
    for (i, result) in reader.records().enumerate() {
        if i >= num_rows {
            break;
        }
        let record = result?;
        println!("{}", record.iter().collect::<Vec<_>>().join(","));
    }

    debug!("Printed {} rows", num_rows);
    Ok(())
}

pub fn load_env_variable(key: &str) -> Result<String> {
    std::env::var(key).context(format!("{} not found in .env file", key))
}

pub fn train_and_test_df(df: &DataFrame) -> Result<(DataFrame, DataFrame)> {
    let total_rows = df.height();
    let train_rows = (total_rows as f64 * 0.7).round() as usize;
    let test_rows = total_rows - train_rows;

    info!("Splitting DataFrame: {} total rows, {} train (70%), {} test (30%)", total_rows, train_rows, test_rows);

    let train_df = df.slice(0, train_rows);
    let test_df = df.slice(train_rows as i64, test_rows);

    Ok((train_df, test_df))
}

pub fn split_features_targets(df: &DataFrame) -> Result<(DataFrame, DataFrame)> {
    let feature_names: Vec<&str> = vec![
        "crim", "zn", "indus", "chas", "nox", "rm", "age", "dis", "rad", "tax",
        "ptratio", "b", "lstat"
    ];
    let target_name: Vec<&str> = vec!["medv"];

    let features = df.select(feature_names)?;
    let target = df.select(target_name)?;

    Ok((features, target))
}

pub fn dataframe_to_dmatrix(features: &DataFrame, targets: &DataFrame) -> Result<DMatrix> {
    info!("Converting Polars DataFrame to DMatrix");

    // Transform Polars DataFrames into 2D arrays in row-major order
    let x_array = features.to_ndarray::<Float64Type>(IndexOrder::C)?;
    let y_array = targets.to_ndarray::<Float64Type>(IndexOrder::C)?;

    // Convert arrays to slices
    let x_slice = x_array.as_slice()
        .ok_or_else(|| anyhow!("Failed to convert features to slice"))?;
    let y_slice = y_array.as_slice()
        .ok_or_else(|| anyhow!("Failed to convert targets to slice"))?;

    // Convert to f32 for XGBoost
    let x_f32: Vec<f32> = x_slice.iter().map(|&v| v as f32).collect();
    let y_f32: Vec<f32> = y_slice.iter().map(|&v| v as f32).collect();

    // Create DMatrix
    let mut dmatrix = DMatrix::from_dense(&x_f32, features.height())?;
    dmatrix.set_labels(&y_f32)?;

    info!("Created DMatrix with shape: {} rows, {} columns", features.height(), features.width());
    Ok(dmatrix)
}

pub fn train_xgboost_model(
    x_train: &DataFrame,
    y_train: &DataFrame,
    x_test: &DataFrame,
    y_test: &DataFrame,
    model_path: &str,
) -> Result<Booster> {
    info!("Starting XGBoost model training");

    // Create DMatrix for training and test sets
    let dmatrix_train = dataframe_to_dmatrix(x_train, y_train)?;
    let dmatrix_test = dataframe_to_dmatrix(x_test, y_test)?;

    info!("Training data: {} rows, {} features", x_train.height(), x_train.width());
    info!("Test data: {} rows, {} features", x_test.height(), x_test.width());

    // Configure XGBoost parameters
    let evaluation_sets = &[(&dmatrix_train, "train"), (&dmatrix_test, "test")];

    let params = parameters::learning::LearningTaskParametersBuilder::default()
        .objective(parameters::learning::Objective::RegLinear)
        .build()
        .map_err(|e| anyhow!("Failed to build learning parameters: {}", e))?;

    let tree_params = parameters::tree::TreeBoosterParametersBuilder::default()
        .max_depth(6)
        .eta(0.3)
        .build()
        .map_err(|e| anyhow!("Failed to build tree parameters: {}", e))?;

    let booster_params = parameters::BoosterParametersBuilder::default()
        .booster_type(parameters::BoosterType::Tree(tree_params))
        .learning_params(params)
        .verbose(true)
        .build()
        .map_err(|e| anyhow!("Failed to build booster parameters: {}", e))?;

    let training_params = parameters::TrainingParametersBuilder::default()
        .dtrain(&dmatrix_train)
        .boost_rounds(100)
        .booster_params(booster_params)
        .evaluation_sets(Some(evaluation_sets))
        .build()
        .map_err(|e| anyhow!("Failed to build training parameters: {}", e))?;

    // Train the model
    info!("Training XGBoost model with 100 rounds...");
    let booster = Booster::train(&training_params)?;

    // Save the model
    info!("Saving model to {}", model_path);
    booster.save(model_path)?;
    info!("Model saved successfully");

    Ok(booster)
}
