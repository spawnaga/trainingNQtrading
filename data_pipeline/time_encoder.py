"""
Cyclical Time Encoder for meaningful temporal representations.

Uses sin/cos encoding to ensure temporal proximity is captured:
- Sunday midnight and Monday 10am have meaningful distance
- Hour 23 and hour 0 are adjacent
- December and January are adjacent
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Tuple


class CyclicalTimeEncoder:
    """
    Encodes temporal features using cyclical sin/cos transformations.

    This ensures that:
    - Hour 23 and Hour 0 are close together
    - Sunday and Monday are close together
    - December and January are close together

    Features generated:
    - hour_sin, hour_cos (24-hour cycle)
    - day_sin, day_cos (7-day cycle)
    - week_sin, week_cos (52-week cycle)
    - month_sin, month_cos (12-month cycle)
    - minute_sin, minute_cos (60-minute cycle)
    - is_market_open (binary)
    - is_session_start (binary)
    - is_session_end (binary)
    - time_to_close (normalized)
    """

    def __init__(
        self,
        session_start: str = "18:00",  # Futures market opens Sunday 6pm ET
        session_end: str = "17:00",     # Closes Friday 5pm ET
        include_session_features: bool = True
    ):
        self.session_start = session_start
        self.session_end = session_end
        self.include_session_features = include_session_features

        # Parse session times
        self.session_start_hour = int(session_start.split(":")[0])
        self.session_start_minute = int(session_start.split(":")[1])
        self.session_end_hour = int(session_end.split(":")[0])
        self.session_end_minute = int(session_end.split(":")[1])

    def _cyclical_encode(
        self,
        values: np.ndarray,
        max_val: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert linear values to cyclical sin/cos representation.

        Args:
            values: Array of values to encode
            max_val: Maximum value in the cycle (e.g., 24 for hours)

        Returns:
            Tuple of (sin_encoded, cos_encoded) arrays
        """
        angle = 2 * np.pi * values / max_val
        return np.sin(angle), np.cos(angle)

    def _is_market_open(self, dt: pd.Timestamp) -> bool:
        """
        Check if market is open at given timestamp.
        NQ futures trade Sunday 6pm - Friday 5pm ET with daily breaks.
        """
        day_of_week = dt.dayofweek  # Monday = 0, Sunday = 6
        hour = dt.hour
        minute = dt.minute

        # Saturday - market closed
        if day_of_week == 5:
            return False

        # Sunday before 6pm - market closed
        if day_of_week == 6 and (hour < self.session_start_hour or
            (hour == self.session_start_hour and minute < self.session_start_minute)):
            return False

        # Friday after 5pm - market closed
        if day_of_week == 4 and (hour > self.session_end_hour or
            (hour == self.session_end_hour and minute >= self.session_end_minute)):
            return False

        # Daily maintenance break (typically 5pm-6pm ET)
        if hour == self.session_end_hour and minute >= self.session_end_minute:
            return False
        if hour == self.session_start_hour and minute < self.session_start_minute:
            return False

        return True

    def _is_session_start(self, dt: pd.Timestamp, window_minutes: int = 30) -> bool:
        """Check if within first N minutes of session."""
        hour = dt.hour
        minute = dt.minute

        if hour == self.session_start_hour:
            minutes_into_session = minute - self.session_start_minute
            if 0 <= minutes_into_session < window_minutes:
                return True
        return False

    def _is_session_end(self, dt: pd.Timestamp, window_minutes: int = 30) -> bool:
        """Check if within last N minutes before session close."""
        hour = dt.hour
        minute = dt.minute

        if hour == self.session_end_hour or (hour == self.session_end_hour - 1 and minute >= 60 - window_minutes):
            minutes_to_close = (self.session_end_hour - hour) * 60 + (self.session_end_minute - minute)
            if 0 < minutes_to_close <= window_minutes:
                return True
        return False

    def _calculate_time_to_close(self, dt: pd.Timestamp) -> float:
        """
        Calculate normalized time until session close.
        Returns value between 0 (just opened) and 1 (about to close).
        """
        hour = dt.hour
        minute = dt.minute

        # Calculate minutes since session start
        if hour >= self.session_start_hour:
            minutes_since_start = (hour - self.session_start_hour) * 60 + (minute - self.session_start_minute)
        else:
            minutes_since_start = (24 - self.session_start_hour + hour) * 60 + (minute - self.session_start_minute)

        # Total session length (roughly 23 hours)
        total_session_minutes = 23 * 60

        return min(minutes_since_start / total_session_minutes, 1.0)

    def encode(self, timestamps: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Encode timestamps into cyclical features.

        Args:
            timestamps: DatetimeIndex of timestamps to encode

        Returns:
            DataFrame with encoded time features
        """
        # Extract time components
        hours = timestamps.hour.values
        minutes = timestamps.minute.values
        days = timestamps.dayofweek.values  # Monday=0, Sunday=6
        weeks = timestamps.isocalendar().week.values
        months = timestamps.month.values

        # Cyclical encoding
        hour_sin, hour_cos = self._cyclical_encode(hours, 24)
        minute_sin, minute_cos = self._cyclical_encode(minutes, 60)
        day_sin, day_cos = self._cyclical_encode(days, 7)
        week_sin, week_cos = self._cyclical_encode(weeks, 52)
        month_sin, month_cos = self._cyclical_encode(months, 12)

        features = {
            'hour_sin': hour_sin,
            'hour_cos': hour_cos,
            'minute_sin': minute_sin,
            'minute_cos': minute_cos,
            'day_sin': day_sin,
            'day_cos': day_cos,
            'week_sin': week_sin,
            'week_cos': week_cos,
            'month_sin': month_sin,
            'month_cos': month_cos,
        }

        # Session-based features
        if self.include_session_features:
            is_open = np.array([self._is_market_open(ts) for ts in timestamps])
            is_start = np.array([self._is_session_start(ts) for ts in timestamps])
            is_end = np.array([self._is_session_end(ts) for ts in timestamps])
            time_to_close = np.array([self._calculate_time_to_close(ts) for ts in timestamps])

            features.update({
                'is_market_open': is_open.astype(float),
                'is_session_start': is_start.astype(float),
                'is_session_end': is_end.astype(float),
                'time_to_close': time_to_close,
            })

        return pd.DataFrame(features, index=timestamps)

    def get_feature_names(self) -> List[str]:
        """Return list of feature names generated by encoder."""
        names = [
            'hour_sin', 'hour_cos',
            'minute_sin', 'minute_cos',
            'day_sin', 'day_cos',
            'week_sin', 'week_cos',
            'month_sin', 'month_cos',
        ]

        if self.include_session_features:
            names.extend([
                'is_market_open',
                'is_session_start',
                'is_session_end',
                'time_to_close',
            ])

        return names

    @property
    def num_features(self) -> int:
        """Return number of features generated."""
        return len(self.get_feature_names())


def compute_time_distance(
    time1: pd.Timestamp,
    time2: pd.Timestamp,
    encoder: CyclicalTimeEncoder
) -> float:
    """
    Compute meaningful distance between two timestamps using cyclical encoding.

    This ensures Sunday midnight and Monday 10am have appropriate distance
    that reflects actual trading time difference.
    """
    enc1 = encoder.encode(pd.DatetimeIndex([time1])).values[0]
    enc2 = encoder.encode(pd.DatetimeIndex([time2])).values[0]

    return np.sqrt(np.sum((enc1 - enc2) ** 2))
