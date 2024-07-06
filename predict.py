from cog import BasePredictor, Path, Input
import torch
import os
from marigold import MarigoldPipeline
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

MODEL_ID = "prs-eth/marigold-v1-0"
MODEL_CACHE = "./marigold-diffuser-cache"

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        if os.path.exists(MODEL_CACHE):
            print(f"Loading model from local directory: {MODEL_CACHE}")
            self.pipe = MarigoldPipeline.from_pretrained(
                MODEL_CACHE, 
                torch_dtype=torch.float16
            ).to("cuda")
        else:
            print(f"Local model not found. Downloading and saving to: {MODEL_CACHE}")
            self.pipe = MarigoldPipeline.from_pretrained(
                MODEL_ID, 
                torch_dtype=torch.float16
            ).to("cuda")
            self.pipe.save_pretrained(MODEL_CACHE)

    def predict(self,
            image: Path = Input(description="Input image"),
            denoising_steps: int = Input(
                ge=1, le=20, default=10,
                description="Denoising steps"
            ),
            ensemble_size: int = Input(
                ge=1, le=20, default=10,
                description="Ensemble size"
            ),
    ) -> Path:
        """Run a single prediction on the model"""
        with torch.no_grad():
            input_image = Image.open(str(image))

            # Predict depth
            pipeline_output = self.pipe(
                input_image,
                denoising_steps=denoising_steps,     # optional
                ensemble_size=ensemble_size,       # optional
                processing_res=768,     # optional
                match_input_res="True",   # optional
                batch_size=0,           # optional
                color_map="Spectral",   # optional
                show_progress_bar=True, # optional
            )

            depth_pred: np.ndarray = pipeline_output.depth_np
            depth_bw_path = "/tmp/depth_bw.png"
            plt.imsave(depth_bw_path, depth_pred, cmap='gray')

            return depth_bw_path
