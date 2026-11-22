from pathlib import Path
from setuptools import find_packages, setup

root = Path(__file__).resolve().parent
assets = [str(p.relative_to(root / 'stage1_gym'))
          for p in (root / 'stage1_gym/assets').rglob('*') if p.is_file()]

setup(
    name='stage1-arm-disturbance-gym',
    version='1.0.0',
    description='PPO locomotion control for a Go2-X5 under arm disturbances',
    packages=find_packages(),
    package_data={'stage1_gym': assets},
    python_requires='>=3.8,<3.9',
    install_requires=['gym==0.26.2', 'numpy==1.23.5'],
    # Isaac Gym and its compatible torch build must be installed separately.
)
