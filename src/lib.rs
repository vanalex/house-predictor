use anyhow::{Context, Result};
use csv::ReaderBuilder;
use log::{debug, info};
use polars::prelude::*;
use std::fs;
use std::io::Write;
use std::path::Path;

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
