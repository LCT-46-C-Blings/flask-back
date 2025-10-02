import os
import sys
import re
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse

import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_score, recall_score, fbeta_score, make_scorer
)
import warnings
warnings.filterwarnings('ignore')

class FHRFeatureExtractor:

    def __init__(self, sampling_rate: float = 5.5):
        self.sampling_rate = sampling_rate
        self.deceleration_threshold = 15.0
        self.acceleration_threshold = 15.0
        self.deceleration_min_duration = 15.0
        self.acceleration_min_duration = 15.0
        self.tachy_threshold = 160.0
        self.brady_threshold = 110.0
        self.tachy_min_duration = 120.0
        self.brady_min_duration = 60.0
        self.low_variability_threshold = 5.0
        self.very_low_variability_threshold = 3.0
        self.high_variability_threshold = 25.0
        self.contraction_threshold_delta = 15.0
        self.contraction_min_duration = 30.0
        self.decel_after_contraction_window = 90.0
        self.nst_required_accelerations = 2

    def _duration_to_samples(self, duration_sec: float) -> int:
        return max(int(duration_sec * self.sampling_rate), 1)

    def _moving_average(self, signal: np.array, window_sec: float) -> np.array:
        window = self._duration_to_samples(window_sec)
        if window % 2 == 0:
            window += 1
        return pd.Series(signal).rolling(window=window, min_periods=1, center=True).mean().to_numpy()

    def _rolling_std(self, signal: np.array, window_sec: float) -> np.array:
        window = self._duration_to_samples(window_sec)
        if window % 2 == 0:
            window += 1
        return pd.Series(signal).rolling(window=window, min_periods=1, center=True).std().to_numpy()

    def _find_segments(self, mask: np.array, min_samples: int) -> List[Tuple[int, int]]:
        segments = []
        start = None
        for idx, flag in enumerate(mask):
            if flag and start is None:
                start = idx
            elif not flag and start is not None:
                if idx - start >= min_samples:
                    segments.append((start, idx))
                start = None
        if start is not None and len(mask) - start >= min_samples:
            segments.append((start, len(mask)))
        return segments

    def _spectral_power_ratio(self, signal: np.array, freq_min: float, freq_max: float) -> float:
        if len(signal) < 16:
            return 0.0
        detrended = signal - np.mean(signal)
        freqs = np.fft.rfftfreq(len(detrended), d=1.0 / self.sampling_rate)
        spectrum = np.abs(np.fft.rfft(detrended)) ** 2
        total_power = np.sum(spectrum[1:])
        if total_power <= 0:
            return 0.0
        band_mask = (freqs >= freq_min) & (freqs <= freq_max)
        band_power = np.sum(spectrum[band_mask])
        return float(band_power / total_power)

    def _detect_contractions(self, uterine_signal: np.array) -> List[Tuple[int, int]]:
        if uterine_signal is None or len(uterine_signal) == 0:
            return []
        baseline = np.median(uterine_signal)
        threshold = baseline + self.contraction_threshold_delta
        mask = uterine_signal >= threshold
        min_samples = self._duration_to_samples(self.contraction_min_duration)
        return self._find_segments(mask, min_samples)

    def _count_decelerations_after_contractions(
        self,
        contraction_segments: List[Tuple[int, int]],
        decel_segments: List[Tuple[int, int]]
    ) -> int:
        if not contraction_segments or not decel_segments:
            return 0
        window_samples = self._duration_to_samples(self.decel_after_contraction_window)
        count = 0
        for c_start, c_end in contraction_segments:
            for d_start, _ in decel_segments:
                if d_start >= c_start and d_start <= c_end + window_samples:
                    count += 1
                    break
        return count

    def extract_features(self, signal: np.array, uterine_signal: Optional[np.array] = None) -> Dict[str, float]:
        if signal is None or len(signal) == 0:
            return self._get_empty_features()
        signal = np.asarray(signal, dtype=float)
        signal = signal[np.isfinite(signal)]
        if len(signal) == 0:
            return self._get_empty_features()

        total_duration_sec = len(signal) / self.sampling_rate if self.sampling_rate > 0 else len(signal)
        baseline_series = self._moving_average(signal, 30.0)
        features: Dict[str, float] = {}

        features['fhr_mean'] = float(np.mean(signal))
        features['fhr_std'] = float(np.std(signal))
        features['fhr_min'] = float(np.min(signal))
        features['fhr_max'] = float(np.max(signal))
        features['fhr_range'] = features['fhr_max'] - features['fhr_min']
        features['fhr_median'] = float(np.median(signal))
        features['fhr_q25'] = float(np.percentile(signal, 25))
        features['fhr_q75'] = float(np.percentile(signal, 75))
        features['fhr_iqr'] = features['fhr_q75'] - features['fhr_q25']

        features['baseline_variability'] = float(np.std(signal))
        rr_intervals = np.diff(signal)
        features['rmssd'] = float(np.sqrt(np.mean(rr_intervals ** 2))) if len(rr_intervals) > 0 else 0.0
        features['variability_coefficient'] = (
            float(features['fhr_std'] / features['fhr_mean']) if features['fhr_mean'] > 0 else 0.0
        )

        short_term_std = self._rolling_std(signal, 10.0)
        long_term_std = self._rolling_std(signal, 60.0)
        short_term_variability = float(np.nanmean(short_term_std)) if short_term_std.size else 0.0
        long_term_variability = float(np.nanmean(long_term_std)) if long_term_std.size else 0.0
        low_variability_ratio = float(np.nanmean(short_term_std < self.low_variability_threshold)) if short_term_std.size else 0.0
        very_low_variability_ratio = float(np.nanmean(short_term_std < self.very_low_variability_threshold)) if short_term_std.size else 0.0
        high_variability_ratio = float(np.nanmean(short_term_std > self.high_variability_threshold)) if short_term_std.size else 0.0
        if np.isnan(low_variability_ratio):
            low_variability_ratio = 0.0
        if np.isnan(very_low_variability_ratio):
            very_low_variability_ratio = 0.0
        if np.isnan(high_variability_ratio):
            high_variability_ratio = 0.0

        features['short_term_variability'] = short_term_variability
        features['long_term_variability'] = long_term_variability
        variability_index_denominator = long_term_variability if long_term_variability > 0 else 1.0
        features['variability_index'] = short_term_variability / variability_index_denominator
        features['low_variability_ratio'] = low_variability_ratio
        features['very_low_variability_ratio'] = very_low_variability_ratio
        features['high_variability_ratio'] = high_variability_ratio
        features['low_variability_flag'] = float(low_variability_ratio > 0.5)
        features['very_low_variability_flag'] = float(very_low_variability_ratio > 0.3)
        features['heart_rate_variability_score'] = short_term_variability

        if len(signal) > 1:
            features['trend_slope'] = float(np.polyfit(range(len(signal)), signal, 1)[0])
        else:
            features['trend_slope'] = 0.0

        decel_mask = signal <= (baseline_series - self.deceleration_threshold)
        decel_segments = self._find_segments(decel_mask, self._duration_to_samples(self.deceleration_min_duration))
        decel_total_seconds = sum((end - start) / self.sampling_rate for start, end in decel_segments)
        decel_depths = [
            float(np.maximum(0.0, np.mean(baseline_series[start:end]) - np.min(signal[start:end])))
            for start, end in decel_segments
        ]
        features['deceleration_count'] = float(len(decel_segments))
        features['deceleration_event_count'] = float(len(decel_segments))
        features['deceleration_total_seconds'] = float(decel_total_seconds)
        features['deceleration_percentage'] = (
            float((decel_total_seconds / total_duration_sec) * 100.0) if total_duration_sec > 0 else 0.0
        )
        features['deceleration_max_drop'] = float(max(decel_depths) if decel_depths else 0.0)

        accel_mask = signal >= (baseline_series + self.acceleration_threshold)
        accel_segments = self._find_segments(accel_mask, self._duration_to_samples(self.acceleration_min_duration))
        accel_total_seconds = sum((end - start) / self.sampling_rate for start, end in accel_segments)
        accel_heights = [
            float(np.maximum(0.0, np.max(signal[start:end]) - np.mean(baseline_series[start:end])))
            for start, end in accel_segments
        ]
        features['acceleration_event_count'] = float(len(accel_segments))
        features['acceleration_total_seconds'] = float(accel_total_seconds)
        features['acceleration_max_height'] = float(max(accel_heights) if accel_heights else 0.0)

        tachy_mask = signal >= self.tachy_threshold
        tachy_segments = self._find_segments(tachy_mask, self._duration_to_samples(self.tachy_min_duration))
        tachy_total_seconds = sum((end - start) / self.sampling_rate for start, end in tachy_segments)
        features['tachycardia_segments'] = float(len(tachy_segments))
        features['tachycardia_total_seconds'] = float(tachy_total_seconds)
        features['has_tachycardia'] = float(len(tachy_segments) > 0)

        brady_mask = signal <= self.brady_threshold
        brady_segments = self._find_segments(brady_mask, self._duration_to_samples(self.brady_min_duration))
        brady_total_seconds = sum((end - start) / self.sampling_rate for start, end in brady_segments)
        features['bradycardia_segments'] = float(len(brady_segments))
        features['bradycardia_total_seconds'] = float(brady_total_seconds)
        features['has_bradycardia'] = float(len(brady_segments) > 0)

        features['sinusoidal_index'] = self._spectral_power_ratio(signal, 0.033, 0.083)
        features['has_sinusoidal_pattern'] = float(
            features['sinusoidal_index'] > 0.45 and 5.0 <= features['fhr_std'] <= 20.0
        )

        diff_signal = np.diff(signal)
        if len(diff_signal) > 1 and np.std(diff_signal) > 0:
            skew_numerator = np.mean((diff_signal - diff_signal.mean()) ** 3)
            skew_denominator = np.std(diff_signal) ** 3 + 1e-6
            sawtooth_index = abs(skew_numerator / skew_denominator) * np.std(diff_signal)
        else:
            sawtooth_index = 0.0
        features['sawtooth_index'] = float(sawtooth_index)
        features['has_sawtooth_pattern'] = float(sawtooth_index > 3.0)

        features['nst_acceleration_count'] = features['acceleration_event_count']
        features['nonstress_reactive'] = float(
            features['acceleration_event_count'] >= self.nst_required_accelerations and
            features['deceleration_event_count'] == 0
        )

        features['absent_variability_flag'] = float(features['very_low_variability_flag'])
        features['decreased_variability_flag'] = float(low_variability_ratio > 0.3)

        uterine_available = 0.0
        contraction_count = 0
        decel_after_contraction = 0
        if uterine_signal is not None and len(uterine_signal) > 0:
            uterine_array = np.asarray(uterine_signal, dtype=float)
            uterine_array = uterine_array[np.isfinite(uterine_array)]
            if len(uterine_array) > 0:
                uterine_available = 1.0
                min_length = min(len(signal), len(uterine_array))
                uterine_array = uterine_array[:min_length]
                decel_segments_adjusted = [
                    (start, min(end, min_length)) for start, end in decel_segments if start < min_length
                ]
                contraction_segments = self._detect_contractions(uterine_array)
                contraction_segments = [
                    (start, min(end, min_length)) for start, end in contraction_segments if start < min_length
                ]
                contraction_count = len(contraction_segments)
                decel_after_contraction = self._count_decelerations_after_contractions(
                    contraction_segments,
                    decel_segments_adjusted
                )

        features['uterine_data_available'] = uterine_available
        features['contraction_count'] = float(contraction_count)
        features['decelerations_after_contractions'] = float(decel_after_contraction)
        ratio = decel_after_contraction / contraction_count if contraction_count > 0 else 0.0
        features['contraction_deceleration_ratio'] = float(ratio)
        features['has_contraction_deceleration_pattern'] = float(decel_after_contraction > 0)
        features['stress_test_positive'] = float(decel_after_contraction >= 2 or ratio >= 0.5)
        features['oxytocin_test_positive'] = float(contraction_count >= 3 and ratio >= 0.4)

        template = self._get_empty_features()
        template.update(features)
        return template

    def _get_empty_features(self) -> Dict[str, float]:
        return {
            'fhr_mean': 0.0,
            'fhr_std': 0.0,
            'fhr_min': 0.0,
            'fhr_max': 0.0,
            'fhr_range': 0.0,
            'fhr_median': 0.0,
            'fhr_q25': 0.0,
            'fhr_q75': 0.0,
            'fhr_iqr': 0.0,
            'baseline_variability': 0.0,
            'rmssd': 0.0,
            'variability_coefficient': 0.0,
            'short_term_variability': 0.0,
            'long_term_variability': 0.0,
            'variability_index': 0.0,
            'low_variability_ratio': 0.0,
            'very_low_variability_ratio': 0.0,
            'high_variability_ratio': 0.0,
            'low_variability_flag': 0.0,
            'very_low_variability_flag': 0.0,
            'heart_rate_variability_score': 0.0,
            'trend_slope': 0.0,
            'deceleration_count': 0.0,
            'deceleration_event_count': 0.0,
            'deceleration_total_seconds': 0.0,
            'deceleration_percentage': 0.0,
            'deceleration_max_drop': 0.0,
            'acceleration_event_count': 0.0,
            'acceleration_total_seconds': 0.0,
            'acceleration_max_height': 0.0,
            'tachycardia_segments': 0.0,
            'tachycardia_total_seconds': 0.0,
            'has_tachycardia': 0.0,
            'bradycardia_segments': 0.0,
            'bradycardia_total_seconds': 0.0,
            'has_bradycardia': 0.0,
            'sinusoidal_index': 0.0,
            'has_sinusoidal_pattern': 0.0,
            'sawtooth_index': 0.0,
            'has_sawtooth_pattern': 0.0,
            'nst_acceleration_count': 0.0,
            'nonstress_reactive': 0.0,
            'uterine_data_available': 0.0,
            'contraction_count': 0.0,
            'decelerations_after_contractions': 0.0,
            'contraction_deceleration_ratio': 0.0,
            'has_contraction_deceleration_pattern': 0.0,
            'stress_test_positive': 0.0,
            'oxytocin_test_positive': 0.0,
            'absent_variability_flag': 0.0,
            'decreased_variability_flag': 0.0
        }

class MedicalHistoryTransformer:
    def __init__(self):
        self.unique_conditions = set()
        self.condition_features = []
        self.is_fitted = False
        
    def clean_condition(self, condition: str) -> str:
        condition = re.sub(r'^[IVX]+\s+', '', condition.strip())
        condition = re.sub(r'\s+', ' ', condition)
        condition = condition.strip(' .,;:')
        condition = condition.lower()
        condition = condition.lower()
        
        if len(condition) < 3:
            return None
            
        return condition
    
    def extract_conditions(self, medical_history: str) -> List[str]:
        if pd.isna(medical_history):
            return []
        
        text = str(medical_history)
        
        if '\n' in text:
            conditions = text.split('\n')
        else:
            conditions = text.split('.')
        
        cleaned_conditions = []
        for condition in conditions:
            cleaned = self.clean_condition(condition)
            if cleaned:
                cleaned_conditions.append(cleaned)
        
        return cleaned_conditions
    
    def fit(self, medical_histories: List[str]):
        print("Extracting unique medical conditions from dataset...")
        
        all_conditions = set()
        condition_counts = {}
        
        for history in medical_histories:
            conditions = self.extract_conditions(history)
            for condition in conditions:
                all_conditions.add(condition)
                condition_counts[condition] = condition_counts.get(condition, 0) + 1
        
        filtered_conditions = {
            condition for condition, count in condition_counts.items() 
            if count >= 2 and len(condition.split()) <= 10
        }
        
        self.unique_conditions = sorted(filtered_conditions)
        
        self.condition_features = [f"is_condition_{i:03d}" for i in range(len(self.unique_conditions))]
        
        self.is_fitted = True
        
        print(f"Found {len(self.unique_conditions)} unique medical conditions")
        
        print("\nExample conditions found:")
        for i, condition in enumerate(self.unique_conditions[:10]):
            print(f"  {self.condition_features[i]}: {condition}")
        if len(self.unique_conditions) > 10:
            print(f"  ... and {len(self.unique_conditions) - 10} more")
    
    def transform_single(self, medical_history: str) -> Dict[str, int]:
        if not self.is_fitted:
            raise ValueError("Transformer must be fitted before transform")
        
        conditions = self.extract_conditions(medical_history)
        
        features = {}
        for i, unique_condition in enumerate(self.unique_conditions):
            feature_name = self.condition_features[i]
            features[feature_name] = int(unique_condition in conditions)
        
        features['total_conditions'] = len(conditions)
        features['text_length'] = len(str(medical_history)) if not pd.isna(medical_history) else 0
        
        return features
    
    def transform(self, medical_histories: List[str]) -> List[Dict[str, int]]:
        return [self.transform_single(history) for history in medical_histories]
    
    def fit_transform(self, medical_histories: List[str]) -> List[Dict[str, int]]:
        self.fit(medical_histories)
        return self.transform(medical_histories)
    
    def get_feature_names(self) -> List[str]:
        return self.condition_features + ['total_conditions', 'text_length']

class HypoxiaPredictor:
    
    def __init__(self, data_path: str = "data"):
        self.data_path = data_path
        self.fhr_extractor = FHRFeatureExtractor()
        self.history_transformer = MedicalHistoryTransformer()
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = None
        
    def load_data(self) -> pd.DataFrame:
        print("Loading data...")
        
        hypoxia_path = os.path.join(self.data_path, "hypoxia.xlsx")
        hypoxia_df = pd.read_excel(hypoxia_path, header=None)
        hypoxia_df.columns = ['folder_id', 'medical_history', 'birth_outcomes'] + [f'col{i}' for i in range(3, len(hypoxia_df.columns))]
        hypoxia_df['label'] = 1
        
        regular_path = os.path.join(self.data_path, "regular.xlsx")
        regular_df = pd.read_excel(regular_path, header=None)
        regular_df.columns = ['folder_id', 'medical_history', 'birth_outcomes'] + [f'col{i}' for i in range(3, len(regular_df.columns))]
        regular_df['label'] = 0
        
        combined_df = pd.concat([
            hypoxia_df[['folder_id', 'medical_history', 'label']], 
            regular_df[['folder_id', 'medical_history', 'label']]
        ], ignore_index=True)
        
        print(f"Loaded {len(hypoxia_df)} hypoxia cases and {len(regular_df)} regular cases")
        return combined_df
    
    def _load_signals_from_folder(self, folder_path: str, subfolder: Optional[str] = None) -> Dict[str, np.array]:
        signals: Dict[str, np.array] = {}
        target_folder = os.path.join(folder_path, subfolder) if subfolder else folder_path
        if not os.path.exists(target_folder):
            return signals

        csv_files = [f for f in os.listdir(target_folder) if f.endswith('.csv')]
        for csv_file in csv_files:
            try:
                file_path = os.path.join(target_folder, csv_file)
                try:
                    signal_data = pd.read_csv(file_path)
                    if 'value' in signal_data.columns:
                        raw_values = signal_data['value'].values
                    elif signal_data.shape[1] >= 2:
                        raw_values = signal_data.iloc[:, 1].values
                    else:
                        raw_values = signal_data.iloc[:, 0].values
                except Exception:
                    signal_data = pd.read_csv(file_path, header=None)
                    raw_values = signal_data.iloc[:, 0].values

                raw_values = pd.to_numeric(raw_values, errors='coerce')
                raw_values = raw_values[np.isfinite(raw_values)]

                if subfolder == 'bpm':
                    raw_values = raw_values[(raw_values >= 50) & (raw_values <= 250)]

                if len(raw_values) <= 10:
                    continue

                base_name = os.path.splitext(csv_file)[0]
                key = re.sub(r'_[0-9]+$', '', base_name)
                signals[key] = raw_values
            except Exception:
                continue
        return signals

    def load_patient_signals(self, folder_path: str) -> List[Tuple[np.array, Optional[np.array]]]:
        combined_signals: List[Tuple[np.array, Optional[np.array]]] = []
        fhr_signals = self._load_signals_from_folder(folder_path, 'bpm')
        uterine_signals = self._load_signals_from_folder(folder_path, 'uterus')
        keys = sorted(set(list(fhr_signals.keys()) + list(uterine_signals.keys())))
        for key in keys:
            combined_signals.append((fhr_signals.get(key), uterine_signals.get(key)))
        return combined_signals

    def load_fhr_signals(self, folder_path: str) -> List[np.array]:
        return list(self._load_signals_from_folder(folder_path, 'bpm').values())

    def _extract_clinical_statistics(self, signal_data: List[Tuple[str, np.array, Optional[np.array]]]) -> Dict:
        all_stats = []
        for signal_name, fhr_signal, uterine_signal in signal_data:
            features = self.fhr_extractor.extract_features(fhr_signal, uterine_signal)
            segments = self._extract_segments_with_timestamps(fhr_signal, uterine_signal)
            stat = {
                'features': features,
                'segments': segments
            }
            all_stats.append(stat)
        
        aggregated = self._aggregate_clinical_statistics(all_stats)
        return aggregated
    
    def _extract_segments_with_timestamps(self, signal: np.array, uterine_signal: Optional[np.array] = None) -> Dict:
        baseline_series = self.fhr_extractor._moving_average(signal, 30.0)
        
        decel_mask = signal <= (baseline_series - self.fhr_extractor.deceleration_threshold)
        decel_min_samples = self.fhr_extractor._duration_to_samples(self.fhr_extractor.deceleration_min_duration)
        decel_segments = self.fhr_extractor._find_segments(decel_mask, decel_min_samples)
        
        accel_mask = signal >= (baseline_series + self.fhr_extractor.acceleration_threshold)
        accel_min_samples = self.fhr_extractor._duration_to_samples(self.fhr_extractor.acceleration_min_duration)
        accel_segments = self.fhr_extractor._find_segments(accel_mask, accel_min_samples)
        
        tachy_mask = signal >= self.fhr_extractor.tachy_threshold
        tachy_min_samples = self.fhr_extractor._duration_to_samples(self.fhr_extractor.tachy_min_duration)
        tachy_segments = self.fhr_extractor._find_segments(tachy_mask, tachy_min_samples)
        
        brady_mask = signal <= self.fhr_extractor.brady_threshold
        brady_min_samples = self.fhr_extractor._duration_to_samples(self.fhr_extractor.brady_min_duration)
        brady_segments = self.fhr_extractor._find_segments(brady_mask, brady_min_samples)
        
        contraction_segments = []
        if uterine_signal is not None and len(uterine_signal) > 0:
            contraction_segments = self.fhr_extractor._detect_contractions(uterine_signal)
        
        def segments_to_timestamps(segments, sampling_rate):
            return [
                {
                    'start_time': float(start / sampling_rate),
                    'end_time': float(end / sampling_rate),
                    'duration': float((end - start) / sampling_rate)
                }
                for start, end in segments
            ]
        
        return {
            'deceleration': segments_to_timestamps(decel_segments, self.fhr_extractor.sampling_rate),
            'acceleration': segments_to_timestamps(accel_segments, self.fhr_extractor.sampling_rate),
            'tachycardia': segments_to_timestamps(tachy_segments, self.fhr_extractor.sampling_rate),
            'bradycardia': segments_to_timestamps(brady_segments, self.fhr_extractor.sampling_rate),
            'contractions': segments_to_timestamps(contraction_segments, self.fhr_extractor.sampling_rate)
        }
    
    def _aggregate_clinical_statistics(self, all_stats: list) -> Dict:
        all_decel_segments = []
        all_accel_segments = []
        all_tachy_segments = []
        all_brady_segments = []
        all_contraction_segments = []
        
        feature_list = []
        
        for stat in all_stats:
            feature_list.append(stat['features'])
            all_decel_segments.extend(stat['segments']['deceleration'])
            all_accel_segments.extend(stat['segments']['acceleration'])
            all_tachy_segments.extend(stat['segments']['tachycardia'])
            all_brady_segments.extend(stat['segments']['bradycardia'])
            all_contraction_segments.extend(stat['segments']['contractions'])
        
        feature_df = pd.DataFrame(feature_list)
        
        stats = {
            'deceleration': {
                'count': len(all_decel_segments),
                'segments': all_decel_segments,
                'total_seconds': sum(s['duration'] for s in all_decel_segments),
                'percentage': float(feature_df['deceleration_percentage'].max()),
                'max_drop_bpm': float(feature_df['deceleration_max_drop'].max())
            },
            'tachycardia': {
                'detected': bool(feature_df['has_tachycardia'].max() > 0),
                'segments_count': len(all_tachy_segments),
                'segments': all_tachy_segments,
                'total_seconds': sum(s['duration'] for s in all_tachy_segments)
            },
            'bradycardia': {
                'detected': bool(feature_df['has_bradycardia'].max() > 0),
                'segments_count': len(all_brady_segments),
                'segments': all_brady_segments,
                'total_seconds': sum(s['duration'] for s in all_brady_segments)
            },
            'heart_rate_variability': {
                'score': float(feature_df['heart_rate_variability_score'].mean()),
                'short_term': float(feature_df['short_term_variability'].mean()),
                'long_term': float(feature_df['long_term_variability'].mean()),
                'low_variability_detected': bool(feature_df['low_variability_flag'].max() > 0),
                'very_low_variability_detected': bool(feature_df['very_low_variability_flag'].max() > 0),
                'absent_variability': bool(feature_df['absent_variability_flag'].max() > 0),
                'decreased_variability': bool(feature_df['decreased_variability_flag'].max() > 0)
            },
            'acceleration': {
                'count': len(all_accel_segments),
                'segments': all_accel_segments,
                'total_seconds': sum(s['duration'] for s in all_accel_segments),
                'max_height_bpm': float(feature_df['acceleration_max_height'].max())
            },
            'sinusoidal_rhythm': {
                'detected': bool(feature_df['has_sinusoidal_pattern'].max() > 0),
                'index': float(feature_df['sinusoidal_index'].max())
            },
            'sawtooth_rhythm': {
                'detected': bool(feature_df['has_sawtooth_pattern'].max() > 0),
                'index': float(feature_df['sawtooth_index'].max())
            },
            'non_stress_test': {
                'reactive': bool(feature_df['nonstress_reactive'].max() > 0),
                'acceleration_count': int(feature_df['nst_acceleration_count'].max())
            },
            'oxytocin_stress_test': {
                'positive': bool(feature_df['oxytocin_test_positive'].max() > 0),
                'stress_test_positive': bool(feature_df['stress_test_positive'].max() > 0),
                'uterine_data_available': bool(feature_df['uterine_data_available'].max() > 0)
            },
            'contractions_and_decelerations': {
                'contraction_count': len(all_contraction_segments),
                'contraction_segments': all_contraction_segments,
                'decelerations_after_contractions': int(feature_df['decelerations_after_contractions'].max()),
                'ratio': float(feature_df['contraction_deceleration_ratio'].max()),
                'pattern_detected': bool(feature_df['has_contraction_deceleration_pattern'].max() > 0)
            }
        }
        
        return stats

    def process_patient_data(self, folder_ids: List[str], category: str) -> Dict[str, float]:
        all_features = {}
        all_signal_data = []
        
        for folder_id in folder_ids:
            folder_path = os.path.join(self.data_path, category, folder_id)
            signal_pairs = self.load_patient_signals(folder_path)

            if signal_pairs:
                patient_features = []
                for fhr_signal, uterine_signal in signal_pairs:
                    if fhr_signal is None or len(fhr_signal) == 0:
                        continue
                    features = self.fhr_extractor.extract_features(fhr_signal, uterine_signal)
                    if features:
                        patient_features.append(features)
                        all_signal_data.append((f'signal_{len(all_signal_data)}', fhr_signal, uterine_signal))

                if patient_features:
                    feature_df = pd.DataFrame(patient_features)
                    for col in feature_df.columns:
                        col_series = feature_df[col].astype(float)
                        all_features[f'{col}_mean'] = float(col_series.mean())
                        all_features[f'{col}_std'] = float(col_series.std(ddof=0))
                        all_features[f'{col}_min'] = float(col_series.min())
                        all_features[f'{col}_max'] = float(col_series.max())
        
        # Add clinical statistics as features (same as inference)
        if all_signal_data:
            clinical_stats = self._extract_clinical_statistics(all_signal_data)
            all_features['clinical_deceleration_count'] = float(clinical_stats.get('deceleration', {}).get('count', 0))
            all_features['clinical_deceleration_percentage'] = float(clinical_stats.get('deceleration', {}).get('percentage', 0))
            all_features['clinical_tachycardia_detected'] = float(clinical_stats.get('tachycardia', {}).get('detected', False))
            all_features['clinical_bradycardia_detected'] = float(clinical_stats.get('bradycardia', {}).get('detected', False))
            all_features['clinical_hrv_score'] = float(clinical_stats.get('heart_rate_variability', {}).get('score', 0))
            all_features['clinical_acceleration_count'] = float(clinical_stats.get('acceleration', {}).get('count', 0))
            all_features['clinical_sinusoidal_detected'] = float(clinical_stats.get('sinusoidal_rhythm', {}).get('detected', False))
            all_features['clinical_sawtooth_detected'] = float(clinical_stats.get('sawtooth_rhythm', {}).get('detected', False))
            all_features['clinical_nst_reactive'] = float(clinical_stats.get('non_stress_test', {}).get('reactive', False))
            all_features['clinical_contraction_count'] = float(clinical_stats.get('contractions_and_decelerations', {}).get('contraction_count', 0))
        
        return all_features
    
    def create_dataset(self) -> pd.DataFrame:
        print("Creating unified dataset...")
        metadata_df = self.load_data()
        
        all_histories = metadata_df['medical_history'].tolist()
        self.history_transformer.fit(all_histories)
        
        processed_data = []
        for idx, row in metadata_df.iterrows():
            folder_ids = str(row['folder_id']).split(',')
            folder_ids = [fid.strip() for fid in folder_ids if fid.strip().isdigit()]
            
            if not folder_ids:
                continue
                
            category = 'hypoxia' if row['label'] == 1 else 'regular'
            
            fhr_features = self.process_patient_data(folder_ids, category)
            
            history_features = self.history_transformer.transform_single(row['medical_history'])
            
            combined_features = {**fhr_features, **history_features, 'label': row['label']}
            processed_data.append(combined_features)
            
            if (idx + 1) % 10 == 0:
                print(f"Processed {idx + 1}/{len(metadata_df)} patients")
        
        dataset = pd.DataFrame(processed_data)
        dataset = dataset.fillna(0)
        
        print(f"Dataset created with {len(dataset)} patients and {len(dataset.columns)-1} features")
        print(f"Medical history features: {len(self.history_transformer.get_feature_names())}")
        
        return dataset
    
    def train_model_with_cv(self, X: np.array, y: np.array, model_type: str = 'rf', 
                           optimize_for_recall: bool = True, beta: float = 2.0, cv_folds: int = 5) -> Dict:
        print(f"Training {model_type} model with {cv_folds}-fold cross-validation...")
        
        if model_type == 'rf':
            if optimize_for_recall:
                estimator = RandomForestClassifier(
                    n_estimators=200, max_depth=15, min_samples_split=3,
                    min_samples_leaf=1, random_state=42,
                    class_weight='balanced_subsample'
                )
            else:
                estimator = RandomForestClassifier(
                    n_estimators=100, max_depth=10, random_state=42
                )
        elif model_type == 'gb':
            estimator = GradientBoostingClassifier(
                n_estimators=100, learning_rate=0.1, max_depth=8, random_state=42
            )
        else:
            estimator = LogisticRegression(
                random_state=42, max_iter=1000, class_weight='balanced'
            )
        self.model = estimator
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', estimator)
        ])

        f_beta_scorer = make_scorer(fbeta_score, beta=beta, zero_division=0)
        
        print("Performing cross-validation...")
        cv_scores = {
            'accuracy': cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy'),
            'recall': cross_val_score(pipeline, X, y, cv=cv, scoring='recall'),
            'precision': cross_val_score(pipeline, X, y, cv=cv, scoring='precision'),
            'f_beta': cross_val_score(pipeline, X, y, cv=cv, scoring=f_beta_scorer),
            'roc_auc': cross_val_score(pipeline, X, y, cv=cv, scoring='roc_auc')
        }
        
        print(f"\n=== CROSS-VALIDATION RESULTS ({cv_folds} folds) ===")
        for metric, scores in cv_scores.items():
            print(f"{metric.upper()}: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
        
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        self.model.fit(X_scaled, y)
        print("Final model training completed")
        
        return {
            'cv_scores': cv_scores,
            'model': self.model,
            'scaler': self.scaler
        }
    
    def save_model(self, model_path: str = "hypoxia_model.pkl"):
        if self.model is None:
            raise ValueError("No trained model to save")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'history_transformer': self.history_transformer,
            'fhr_extractor': self.fhr_extractor
        }
        
        joblib.dump(model_data, model_path)
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path: str = "hypoxia_model.pkl"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        try:
            model_data = joblib.load(model_path)
        except AttributeError as e:
            if "__main__" in str(e) and "MedicalHistoryTransformer" in str(e):
                print("Warning: Model was saved from __main__, attempting compatibility fix...")
                # Temporarily add our classes to __main__ so joblib can find them
                import __main__
                __main__.MedicalHistoryTransformer = MedicalHistoryTransformer
                __main__.FHRFeatureExtractor = FHRFeatureExtractor
                __main__.HypoxiaPredictor = HypoxiaPredictor
                # Try loading again
                model_data = joblib.load(model_path)
            else:
                raise
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.history_transformer = model_data['history_transformer']
        self.fhr_extractor = model_data['fhr_extractor']
        
        print(f"Model loaded from {model_path}")
    
    def predict_single_patient(self, folder_ids: List[str], category: str, medical_history: str) -> Dict:
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        fhr_features = self.process_patient_data(folder_ids, category)
        
        history_features = self.history_transformer.transform_single(medical_history)
        
        all_features = {**fhr_features, **history_features}
        
        feature_vector = []
        for feature_name in self.feature_names:
            feature_vector.append(all_features.get(feature_name, 0))
        
        feature_vector = np.array(feature_vector).reshape(1, -1)
        
        feature_vector_scaled = self.scaler.transform(feature_vector)
        
        prediction = self.model.predict(feature_vector_scaled)[0]
        probability = self.model.predict_proba(feature_vector_scaled)[0, 1]
        
        return {
            'prediction': prediction,
            'probability': probability,
            'confidence': probability if prediction == 1 else 1 - probability,
            'features_used': len(self.feature_names)
        }

    def evaluate_model(self, X_test: np.array, y_test: np.array, beta: float = 2.0) -> Dict:
        X_test_scaled = self.scaler.transform(X_test)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        y_pred_standard = (y_pred_proba >= 0.5).astype(int)
        
        thresholds = np.arange(0.1, 0.9, 0.05)
        best_fbeta = 0
        best_threshold = 0.5
        
        for threshold in thresholds:
            y_pred_thresh = (y_pred_proba >= threshold).astype(int)
            fbeta = fbeta_score(y_test, y_pred_thresh, beta=beta, zero_division=0)
            if fbeta > best_fbeta:
                best_fbeta = fbeta
                best_threshold = threshold
        
        y_pred_optimized = (y_pred_proba >= best_threshold).astype(int)
        
        results = {
            'auc': roc_auc_score(y_test, y_pred_proba),
            'recall_standard': recall_score(y_test, y_pred_standard),
            'recall_optimized': recall_score(y_test, y_pred_optimized),
            'precision_standard': precision_score(y_test, y_pred_standard, zero_division=0),
            'precision_optimized': precision_score(y_test, y_pred_optimized, zero_division=0),
            'fbeta_standard': fbeta_score(y_test, y_pred_standard, beta=beta, zero_division=0),
            'fbeta_optimized': fbeta_score(y_test, y_pred_optimized, beta=beta, zero_division=0),
            'best_fbeta': max(fbeta_score(y_test, y_pred_standard, beta=beta, zero_division=0),
                            fbeta_score(y_test, y_pred_optimized, beta=beta, zero_division=0)),
            'best_threshold': best_threshold
        }
        
        print(f"\n=== MODEL EVALUATION (Beta={beta}) ===")
        print(f"AUC Score: {results['auc']:.4f}")
        print(f"Standard Threshold (0.5): Recall={results['recall_standard']:.3f}, F-beta={results['fbeta_standard']:.3f}")
        print(f"Optimized Threshold ({best_threshold:.2f}): Recall={results['recall_optimized']:.3f}, F-beta={results['fbeta_optimized']:.3f}")
        
        return results
    
    def run_pipeline(self, test_size: float = 0.2, model_type: str = 'rf', 
                    beta: float = 2.0, optimize_for_recall: bool = True, 
                    save_model: bool = True, model_path: str = "hypoxia_model.pkl",
                    cv_folds: int = 5) -> Dict:
        print("=== FETAL HYPOXIA PREDICTION PIPELINE ===")
        
        dataset = self.create_dataset()
        
        feature_columns = [col for col in dataset.columns if col != 'label']
        X = dataset[feature_columns].values
        y = dataset['label'].values
        self.feature_names = feature_columns
        
        print(f"Feature matrix shape: {X.shape}")
        print(f"Class distribution: {np.bincount(y)}")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"Train set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        cv_results = self.train_model_with_cv(X_train, y_train, model_type, optimize_for_recall, beta, cv_folds)
        
        test_results = self.evaluate_model(X_test, y_test, beta)
        
        if save_model:
            self.save_model(model_path)
        
        return {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'history_transformer': self.history_transformer,
            'cv_results': cv_results,
            'test_results': test_results,
            'dataset': dataset
        }

def main():
    parser = argparse.ArgumentParser(description='Fetal Hypoxia Training System')
    parser.add_argument('--data-path', default='data', help='Path to data directory')
    parser.add_argument('--model', default='rf', choices=['rf', 'gb', 'lr'], help='Model type')
    parser.add_argument('--beta', type=float, default=2.0, help='Beta for F-beta score')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set size')
    parser.add_argument('--cv-folds', type=int, default=5, help='Cross-validation folds')
    parser.add_argument('--model-path', default='hypoxia_model.pkl', help='Path to save trained model')
    parser.add_argument('--no-save', action='store_true', help='Do not save the trained model')
    
    args = parser.parse_args()
    
    predictor = HypoxiaPredictor(args.data_path)
    
    results = predictor.run_pipeline(
        test_size=args.test_size,
        model_type=args.model,
        beta=args.beta,
        optimize_for_recall=True,
        save_model=not args.no_save,
        model_path=args.model_path,
        cv_folds=args.cv_folds
    )
    
    print("\n" + "="*50)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("="*50)
    
    cv_results = results['cv_results']['cv_scores']
    test_results = results['test_results']
    
    print(f"\nCROSS-VALIDATION RESULTS:")
    print(f"  F-beta (β={args.beta}): {cv_results['f_beta'].mean():.3f} (±{cv_results['f_beta'].std():.3f})")
    print(f"  Recall: {cv_results['recall'].mean():.3f} (±{cv_results['recall'].std():.3f})")
    print(f"  Precision: {cv_results['precision'].mean():.3f} (±{cv_results['precision'].std():.3f})")
    print(f"  AUC: {cv_results['roc_auc'].mean():.3f} (±{cv_results['roc_auc'].std():.3f})")
    
    print(f"\nTEST SET RESULTS:")
    print(f"  Best F-beta Score: {test_results['best_fbeta']:.3f}")
    print(f"  Recall (Sensitivity): {test_results['recall_optimized']:.3f}")
    print(f"  AUC Score: {test_results['auc']:.3f}")
    print(f"  Optimal Threshold: {test_results['best_threshold']:.3f}")
    
    print(f"\nMODEL FEATURES:")
    print(f"  Total features: {len(results['feature_names'])}")
    print(f"  Medical history features: {len(results['history_transformer'].get_feature_names())}")
    print(f"  Medical conditions identified: {len(results['history_transformer'].unique_conditions)}")
    
    if not args.no_save:
        print(f"\nModel saved to: {args.model_path}")
        print("Use inference.py for making predictions with the trained model")

if __name__ == "__main__":
    main()