use anyhow::Result;
use csv::ReaderBuilder;
use std::fs;
use std::io::Write;
use std::path::Path;

fn download_csv(url: &str, file_path: &str) -> Result<()> {
    let response = reqwest::blocking::get(url)?;
    let content = response.bytes()?;

    // Create data folder if it doesn't exist
    if let Some(parent) = Path::new(file_path).parent() {
        fs::create_dir_all(parent)?;
    }

    let mut file = fs::File::create(file_path)?;
    file.write_all(&content)?;

    println!("Downloaded CSV to {}", file_path);
    Ok(())
}

fn print_csv_rows(file_path: &str, num_rows: usize) -> Result<()> {
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

    Ok(())
}

fn main() -> Result<()> {
    let url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv";
    let file_path = "data/BostonHousing.csv";

    download_csv(url, file_path)?;
    println!("\nFirst 10 rows:");
    print_csv_rows(file_path, 10)?;

    Ok(())
}
