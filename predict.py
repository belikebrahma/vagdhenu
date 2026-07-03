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

# Mock wandb entirely — Cog pins Pydantic v1 but wandb requires Pydantic v2.
# We never use wandb for inference, so we intercept all wandb.* imports.
import types
import importlib
import importlib.abc
import importlib.machinery

class _FakeModule(types.ModuleType):
    """A fake module that returns itself for any attribute access."""
    def __init__(self, name):
        super().__init__(name)
        self.__path__ = ['/fake/wandb']
        self.__package__ = name
        self.__all__ = []
        self.__spec__ = importlib.machinery.ModuleSpec(name, None, is_package=True)
    def __getattr__(self, name):
        fullname = f'{self.__name__}.{name}'
        if fullname not in sys.modules:
            sys.modules[fullname] = _FakeModule(fullname)
        return sys.modules[fullname]

class _WandbFinder(importlib.abc.MetaPathFinder):
    """Intercept any import of wandb or wandb.* and return a _FakeModule."""
    def find_spec(self, fullname, path, target=None):
        if fullname == 'wandb' or fullname.startswith('wandb.'):
            return importlib.machinery.ModuleSpec(fullname, self, is_package=True)
        return None
    def create_module(self, spec):
        return _FakeModule(spec.name)
    def exec_module(self, module):
        pass

sys.meta_path.insert(0, _WandbFinder())

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
