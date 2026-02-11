use anyhow::{Context, Result};
use csv::ReaderBuilder;
use log::{info, debug};
use std::env;
use std::fs;
use std::io::Write;
use std::path::Path;

fn download_csv(url: &str, file_path: &str) -> Result<()> {
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

fn print_csv_rows(file_path: &str, num_rows: usize) -> Result<()> {
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

fn main() -> Result<()> {
    env_logger::init();
    dotenv::dotenv().ok();

    let url = env::var("CSV_URL").context("CSV_URL not found in .env file")?;
    let file_path = "data/BostonHousing.csv";

    if Path::new(file_path).exists() {
        info!("File already exists at {}, skipping download", file_path);
    } else {
        download_csv(&url, file_path)?;
    }

    info!("Displaying first 10 rows:");
    print_csv_rows(file_path, 10)?;

    Ok(())
}
