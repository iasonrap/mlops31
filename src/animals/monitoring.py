import pandas as pd
from evidently.metrics import (
    ClassificationClassBalance,
    ClassificationConfusionMatrix,
    ClassificationQualityByClass,
    ClassificationQualityMetric,
)
from evidently.report import Report


def run_analysis(evaluation_results: pd.DataFrame, predictions: pd.DataFrame) -> None:
     # Prepare data for Evidently
    # Add the predicted class column based on maximum probability
    evaluation_results["prediction"] = evaluation_results.iloc[:, 1:].idxmax(axis=1)  # From probabilities
    predictions["prediction"] = predictions.iloc[:, 1:].idxmax(axis=1)

    # Ensure the "target" column is present in both dataframes
    reference_data = evaluation_results[["target", "prediction"]]
    print(reference_data.head(5))
    current_data = predictions[["target", "prediction"]]
    print(current_data.head(5))
    
    # Create the Classification Performance Report
    report = Report(metrics=[ClassificationQualityMetric(), ClassificationClassBalance(), ClassificationConfusionMatrix(),
                             ClassificationQualityByClass()])
    report.run(reference_data=reference_data, current_data=current_data)

    # Save the report as an HTML file
    report.save_html("reports/classification_performance_report.html")

if __name__ == "__main__":
    # Load reference and current data
    reference_data = pd.read_csv("reports/evaluation_results.csv")
    current_data = pd.read_csv("reports/predictions.csv")

    # Run drift analysis
    run_analysis(reference_data, current_data)