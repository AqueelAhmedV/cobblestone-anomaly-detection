import numpy as np
import matplotlib.pyplot as plt
from streamad.util import UnivariateDS
from streamad.model import SpotDetector
from generate_stream import generate_data_stream
from scipy.signal import savgol_filter
from typing import Tuple, List

def run_anomaly_detection(n_points: int = 10000, window_size: int = 5, q: float = 1e-4) -> Tuple[UnivariateDS, List[float]]:
    """
    Run anomaly detection on a generated data stream using the SPOT algorithm.

    Note: window_size=5 with q=1e-4 has been observed to give the best F1 score.

    Args:
        n_points (int): Number of data points to process. Default is 10000.
        window_size (int): Size of the sliding window for smoothing. Default is 5.
        q (float): False alarm probability for the SPOT detector. Default is 1e-4.

    Returns:
        Tuple[UnivariateDS, List[float]]: A tuple containing:
            - UnivariateDS: Dataset with the generated data and labels.
            - List[float]: List of anomaly scores.
    """
    stream = generate_data_stream()
    ds = UnivariateDS()
    model = SpotDetector(prob=q)
    
    scores = []
    data_list, label_list = [], []
    window = []
    
    for _ in range(n_points):
        x, is_anomaly = next(stream)
        data_list.append(x)
        label_list.append(is_anomaly)
        
        # Maintain a sliding window of data points
        window.append(x)
        if len(window) > window_size:
            window.pop(0)
        
        if len(window) == window_size:
            # Apply Savitzky-Golay filter for smoothing
            smoothed_window = savgol_filter(window, window_size, 3)
            score = model.fit_score(np.array([smoothed_window[-1]]))
            scores.append(score if score is not None else 0)
        else:
            scores.append(0)

    # Populate UnivariateDS object
    ds.data = np.array(data_list).reshape(-1, 1)
    ds.label = np.array(label_list)
    ds.date = np.arange(n_points)
    ds.features = ['value']

    return ds, scores

def evaluate_model(ds: UnivariateDS, scores: List[float]) -> Tuple[float, float, float]:
    """
    Evaluate the performance of the anomaly detection model.

    Args:
        ds (UnivariateDS): Dataset containing the true labels.
        scores (List[float]): List of anomaly scores.

    Returns:
        Tuple[float, float, float]: Precision, Recall, and F1 score.
    """
    threshold = np.percentile(scores, 95)
    detected_anomalies = np.array(scores) > threshold
    
    true_positives = np.sum(detected_anomalies & ds.label)
    false_positives = np.sum(detected_anomalies & ~ds.label)
    false_negatives = np.sum(~detected_anomalies & ds.label)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1_score

def plot_results(ds: UnivariateDS, scores: List[float]) -> None:
    """
    Plot the results of the anomaly detection.

    Args:
        ds (UnivariateDS): Dataset containing the data and true labels.
        scores (List[float]): List of anomaly scores.
    """
    threshold = np.percentile(scores, 95)
    detected_anomalies = np.array(scores) > threshold

    plt.figure(figsize=(12, 6))
    plt.plot(ds.date, ds.data, label='Data', color='gray', alpha=0.5)
    plt.plot(ds.date, scores, label='Anomaly Scores', color='green', alpha=0.5)
    
    plt.scatter(ds.date[detected_anomalies], ds.data[detected_anomalies], 
                color='blue', label='Detected Anomalies', zorder=5)
    plt.scatter(ds.date[ds.label], ds.data[ds.label], 
                color='red', label='Actual Anomalies', zorder=10)

    plt.legend()
    plt.title('SPOT Anomaly Detection')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.show()

if __name__ == "__main__":
    # Run anomaly detection
    ds, scores = run_anomaly_detection()

    # Evaluate the model
    precision, recall, f1_score = evaluate_model(ds, scores)
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1_score:.4f}")

    # Plot the results
    plot_results(ds, scores)
