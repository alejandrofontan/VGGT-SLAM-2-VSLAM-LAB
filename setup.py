from setuptools import setup, find_packages

setup(
    name='vggt_slam',
    version='2.0.0',
    description='A feedforward SLAM system optimized on the SL(4) manifold.',
    author='Dominic Maggio',
    packages=find_packages(include=['evals', 'evals.*', 'vggt_slam', 'vggt_slam.*']),

    # VSLAM-LAB entry point (Baselines/baseline_files/baseline_vggtslam.py in VSLAM-LAB)
    py_modules=['vslamlab_vggtslam_mono'],
    entry_points={
        'console_scripts': [
            'vslamlab_vggtslam_mono=vslamlab_vggtslam_mono:main',
        ],
    },
)
