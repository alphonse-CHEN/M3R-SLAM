from pathlib import Path
from setuptools import setup

asmk = Path(__file__).parent / "asmk"
setup(
    install_requires=[
        "scikit-learn",
        "roma",
        "gradio",
        "matplotlib",
        "tqdm",
        "opencv-python",
        "scipy",
        "einops",
        "trimesh",
        "tensorboard",
        "pyglet",
        "huggingface-hub[torch]>=0.22",
        "curope",
        f"asmk @ {asmk.as_uri()}",
    ],
)
