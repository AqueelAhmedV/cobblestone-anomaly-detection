import random
import math
from typing import Generator, Tuple

def generate_data_stream(
    base_value: float = 100,
    trend_strength: float = 0.1,
    seasonal_strength: float = 10,
    noise_level: float = 5,
    anomaly_probability: float = 0.01,
    anomaly_scale: float = 3
) -> Generator[Tuple[float, bool], None, None]:
    """
    Generate a continuous data stream with regular patterns, seasonal elements, random noise, and anomalies.

    This generator function creates a synthetic data stream that simulates real-world time series data.
    It incorporates several components:
    1. A base value
    2. An upward trend
    3. A seasonal pattern (sine wave)
    4. Random noise
    5. Occasional anomalies

    Args:
        base_value (float): The starting value for the data stream. Defaults to 100.
        trend_strength (float): The strength of the upward trend. Higher values create a steeper trend. Defaults to 0.1.
        seasonal_strength (float): The amplitude of the seasonal pattern. Higher values create more pronounced seasonality. Defaults to 10.
        noise_level (float): The maximum amount of random noise to add to each data point. Defaults to 5.
        anomaly_probability (float): The probability of generating an anomaly (0-1). Defaults to 0.01 (1% chance).
        anomaly_scale (float): The scale factor for anomalies. Anomalies will be multiplied by a random value between this and (this + 1). Defaults to 3.

    Yields:
        Tuple[float, bool]: A tuple containing:
            - float: The next value in the data stream.
            - bool: A flag indicating whether this point is an anomaly (True) or not (False).

    Example:
        >>> stream = generate_data_stream(base_value=100, trend_strength=0.1, seasonal_strength=10)
        >>> for _ in range(5):
        ...     value, is_anomaly = next(stream)
        ...     print(f"{value:.2f} {'(Anomaly)' if is_anomaly else ''}")
        100.23
        101.56
        102.89
        104.12 (Anomaly)
        105.45
    """
    t = 0
    while True:
        # Generate base pattern (trend + seasonality)
        value = base_value + trend_strength * t + seasonal_strength * math.sin(t / 50)
        
        # Add random noise
        value += random.uniform(-noise_level, noise_level)
        
        # Determine if this point is an anomaly
        is_anomaly = random.random() < anomaly_probability
        
        # If it's an anomaly, scale the value
        if is_anomaly:
            value *= random.uniform(anomaly_scale, anomaly_scale + 1)
        
        yield value, is_anomaly
        t += 1

# Example usage:
# if __name__ == "__main__":
#     stream = generate_data_stream()
#     for i, (value, is_anomaly) in enumerate(stream):
#         print(f"Data point {i}: {value:.2f} {'(Anomaly)' if is_anomaly else ''}")
#         if i >= 10:  # Stop after 10 points for this example
#             break
