import os
import sys
import torch
import numpy as np
import soundfile as sf
from cog import BasePredictor, Input, Path

# Add src/ and BigVGAN/ folders to Python path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
sys.path.append("/app/BigVGAN")

# Setup environment variables so api.py loads properly
os.environ["VAGDHENU_DEVICE"] = "cuda"
os.environ["CHAMP_ROOT"] = "/app/models"

from api import TTSModels, infer_process, SR

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.models = TTSModels()
        self.models.load(device="cuda")

    def predict(
        self,
        ref_audio: Path = Input(description="Reference audio file (WAV format recommended)"),
        ref_text: str = Input(description="Reference text transcribing the reference audio"),
        gen_text: str = Input(description="Sanskrit text to generate (pre-processed and sandhi-converted)"),
        speed: float = Input(description="Pace of generation (default 1.0)", default=1.0),
        nfe_step: int = Input(description="Integration steps (default 64)", default=64),
        cfg_strength: float = Input(description="Classifier-free guidance strength (default 2.0)", default=2.0),
        fix_duration: float = Input(description="Explicit duration in seconds", default=None),
    ) -> Path:
        """Run a single prediction on the model"""
        ref_audio_str = str(ref_audio)
        
        # Run core inference
        w, sr, _ = infer_process(
            ref_audio_str, ref_text, gen_text, self.models.cfm, self.models.cap,
            mel_spec_type="vocos", speed=speed,
            nfe_step=nfe_step, cfg_strength=cfg_strength,
            device="cuda", fix_duration=fix_duration
        )
        
        # Decode using BigVGAN
        y = self.models.bvgan_decode(self.models.cap.last)
        
        # Normalize
        mx = np.abs(y).max()
        if mx > 1.0:
            y = y / mx * 0.97
            
        # Write to a temporary output path
        out_path = Path("/tmp/out.wav")
        sf.write(str(out_path), y, SR, format="WAV")
        
        return out_path
