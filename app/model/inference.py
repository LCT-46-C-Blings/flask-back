import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, Optional
import argparse
import json

sys.path.append(os.path.dirname(__file__))

from hypoxia_predictor import HypoxiaPredictor, FHRFeatureExtractor, MedicalHistoryTransformer

class HypoxiaInference:
    def __init__(self, model_path: str = "hypoxia_model.pkl"):
        self.predictor = HypoxiaPredictor()
        self.predictor.load_model(model_path)
        self.fhr_extractor = self.predictor.fhr_extractor
        print("Inference system ready!")
    
    def _extract_clinical_statistics(self, signal_data: list) -> Dict:
        all_stats = []
        
        for signal_name, fhr_signal, uterine_signal in signal_data:
            features = self.fhr_extractor.extract_features(fhr_signal, uterine_signal)
            segments = self._extract_segments_with_timestamps(fhr_signal, uterine_signal)
            
            stat = {
                'signal_name': signal_name,
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
    
    def predict_from_arrays(
        self,
        fhr_timestamps: np.ndarray,
        fhr_values: np.ndarray,
        uc_timestamps: Optional[np.ndarray] = None,
        uc_values: Optional[np.ndarray] = None,
        blood_gases: Optional[Dict[str, float]] = None,
        medical_history: str = ""
    ) -> Dict:
        try:
            print(f"\n{'='*80}")
            print(f"Processing Signal Data")
            print(f"{'='*80}")
            
            fhr_values = np.array(fhr_values, dtype=float)
            fhr_values = fhr_values[np.isfinite(fhr_values)]
            fhr_values = fhr_values[(fhr_values >= 50) & (fhr_values <= 250)]
            
            if len(fhr_values) <= 10:
                return {'error': 'Insufficient FHR data points'}
            
            print(f"FHR signal: {len(fhr_values)} data points")
            
            uterine_signal = None
            if uc_timestamps is not None and uc_values is not None:
                uc_values = np.array(uc_values, dtype=float)
                uc_values = uc_values[np.isfinite(uc_values)]
                if len(uc_values) > 10:
                    uterine_signal = uc_values
                    print(f"Uterine signal: {len(uc_values)} data points")
            
            if blood_gases:
                print(f"Blood gas data provided: {list(blood_gases.keys())}")
            
            print("\nExtracting features for prediction...")
            features = self.fhr_extractor.extract_features(fhr_values, uterine_signal)
            
            if not features:
                return {'error': 'Could not extract features'}
            
            patient_features = [features]
            signal_data = [('signal_1', fhr_values, uterine_signal)]
            
            if not patient_features:
                return {'error': 'Could not extract features'}
            
            # Calculate clinical statistics
            clinical_stats = self._extract_clinical_statistics(signal_data)
            
            feature_df = pd.DataFrame(patient_features)
            all_features = {}
            for col in feature_df.columns:
                col_series = feature_df[col].astype(float)
                all_features[f'{col}_mean'] = float(col_series.mean())
                all_features[f'{col}_std'] = float(col_series.std(ddof=0))
                all_features[f'{col}_min'] = float(col_series.min())
                all_features[f'{col}_max'] = float(col_series.max())
            
            # Add clinical statistics as additional features
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
            
            history_features = self.predictor.history_transformer.transform_single(medical_history)
            all_features.update(history_features)
            
            feature_vector = []
            for feature_name in self.predictor.feature_names:
                feature_vector.append(all_features.get(feature_name, 0))
            
            feature_vector = np.array(feature_vector).reshape(1, -1)
            
            feature_vector_scaled = self.predictor.scaler.transform(feature_vector)
            prediction = self.predictor.model.predict(feature_vector_scaled)[0]
            probability = self.predictor.model.predict_proba(feature_vector_scaled)[0, 1]
            confidence = probability if prediction == 1 else 1 - probability
            
            if prediction == 1:
                risk_level = "HIGH RISK"
                if confidence >= 0.8:
                    recommendation = "IMMEDIATE MEDICAL ATTENTION REQUIRED"
                elif confidence >= 0.6:
                    recommendation = "INCREASED MONITORING RECOMMENDED"
                else:
                    recommendation = "CONTINUE MONITORING WITH CAUTION"
            else:
                risk_level = "LOW RISK"
                recommendation = "CONTINUE STANDARD MONITORING"
            
            result = {
                'hypoxia_probability': float(probability),
                'clinical_statistics': clinical_stats
            }
            
            print(f"\nPrediction complete:")
            print(f"  Hypoxia Probability: {probability:.3f}")
            
            return result
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {'error': str(e)}

def main():
    parser = argparse.ArgumentParser(
        description='Fetal Hypoxia Inference System - Array Input',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python inference.py --fhr-timestamps "[1.0,2.0,3.0]" --fhr-values "[120,125,130]"
  
  python inference.py --fhr-timestamps "[1.0,2.0]" --fhr-values "[120,125]" --uc-timestamps "[1.0,2.0]" --uc-values "[10,15]"
        """
    )
    
    parser.add_argument('--fhr-timestamps', required=True, help='FHR timestamps as JSON array of floats')
    parser.add_argument('--fhr-values', required=True, help='FHR values (BPM) as JSON array of floats')
    parser.add_argument('--uc-timestamps', help='Uterine contraction timestamps as JSON array of floats (optional)')
    parser.add_argument('--uc-values', help='Uterine contraction values as JSON array of floats (optional)')
    parser.add_argument('--blood-gases', help='Blood gas measurements as JSON dict (optional)')
    parser.add_argument('--medical-history', default='', help='Patient medical history text')
    parser.add_argument('--model-path', default='hypoxia_model.pkl', help='Path to trained model (default: hypoxia_model.pkl)')
    
    args = parser.parse_args()
    
    try:
        fhr_timestamps = np.array(json.loads(args.fhr_timestamps), dtype=float)
        fhr_values = np.array(json.loads(args.fhr_values), dtype=float)
        
        uc_timestamps = None
        uc_values = None
        if args.uc_timestamps and args.uc_values:
            uc_timestamps = np.array(json.loads(args.uc_timestamps), dtype=float)
            uc_values = np.array(json.loads(args.uc_values), dtype=float)
        
        blood_gases = None
        if args.blood_gases:
            blood_gases = json.loads(args.blood_gases)
        
        inference = HypoxiaInference(args.model_path)
        
        result = inference.predict_from_arrays(
            fhr_timestamps=fhr_timestamps,
            fhr_values=fhr_values,
            uc_timestamps=uc_timestamps,
            uc_values=uc_values,
            blood_gases=blood_gases,
            medical_history=args.medical_history
        )
        
        if 'error' in result:
            print(json.dumps(result, indent=2))
            return 1
        
        # Print result as JSON dictionary
        print(json.dumps(result, indent=2))
        
        return 0
        
    except Exception as e:
        print(f"\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
