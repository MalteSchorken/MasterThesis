import librosa
import numpy as np
from .fitness import compute_audio_feature

class Target():
    
    # -------------------------
    # Constructor
    # -------------------------
    def __init__(self, y, onsets=None, objectives=[("stft", "cosh")]):
        self.y = y
        self.onsets = onsets if onsets is not None else self.detect_onsets()
    
        # Storage:
        # feature_per_snippet[onset][feature_name] = feature_tensor
        self.feature_per_snippet = {}
    
        # Compute all needed features
        feature_names = list({obj[0] for obj in objectives})
        self.calc_features_for_snippets(feature_names)


    def detect_onsets(self):
        y = librosa.resample(y=self.y, orig_sr=22050, target_sr=11025)
        onset_frames = librosa.onset.onset_detect(y=y, units='frames')
        oenv = librosa.onset.onset_strength(y=y)
        backtracked_onset_frames = librosa.onset.onset_backtrack(onset_frames, oenv)
        return librosa.frames_to_samples(backtracked_onset_frames)


    def calc_features_for_snippets(self, feature_names):
        for i, onset in enumerate(self.onsets):
            if i + 1 < len(self.onsets):
                next_onset = self.onsets[i+1]
                snippet = self.y[onset:next_onset]
            else:
                snippet = self.y[onset:]
            
            self.feature_per_snippet[onset] = {}
    
            for feature_name in feature_names:
                feature = compute_audio_feature(snippet, feature_name)
                self.feature_per_snippet[onset][feature_name] = feature