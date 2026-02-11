use anyhow::Result;
use house_predictor::*;
use log::info;
use std::path::Path;

fn main() -> Result<()> {
    env_logger::init();
    dotenv::dotenv().ok();

    let url = load_env_variable("CSV_URL")?;
    let file_path = "data/BostonHousing.csv";

    if Path::new(file_path).exists() {
        info!("File already exists at {}, skipping download", file_path);
    } else {
        download_csv(&url, file_path)?;
    }

    info!("Displaying first 10 rows:");
    print_csv_rows(file_path, 10)?;

    let df = load_csv_to_dataframe(file_path)?;
    info!("\nDataFrame head:\n{}", df.head(Some(10)));

    let (train_df, test_df) = train_and_test_df(&df)?;
    info!("\nTrain DataFrame shape: {:?}", train_df.shape());
    info!("Train DataFrame:\n{}", train_df.head(Some(5)));
    info!("\nTest DataFrame shape: {:?}", test_df.shape());
    info!("Test DataFrame:\n{}", test_df.head(Some(5)));

    let (features, target) = split_features_targets(&train_df)?;
    info!("\nFeatures DataFrame shape: {:?}", features.shape());
    info!("Target DataFrame shape: {:?}", target.shape());

    let (features, target) = split_features_targets(&test_df)?;
    info!("\nFeatures DataFrame shape: {:?}", features.shape());
    info!("Target DataFrame shape: {:?}", target.shape());

    Ok(())
}
