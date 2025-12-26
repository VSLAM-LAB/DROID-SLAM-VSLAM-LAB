from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, include_paths, library_paths
import os.path as osp
import torch
import os

ROOT = osp.dirname(osp.abspath(__file__))
torch_include_dirs = include_paths()
torch_library_dirs = library_paths()

setup(
    name='vslamlab_droidslam',
    version='0.1',
    description='DROID-SLAM',
    package_data={
        'droid_slam.configs': ['*.yaml'], 
    },
    include_package_data=True,
    py_modules=['vslamlab_droidslam_mono', 'vslamlab_droidslam_rgbd', 'vslamlab_droidslam_stereo'],
    packages=find_packages(where='.'),
    package_dir={
        'droid_slam': 'droid_slam',
    },
    entry_points={
        'console_scripts': [
            'vslamlab_droidslam_mono = vslamlab_droidslam_mono:main',
            'vslamlab_droidslam_rgbd = vslamlab_droidslam_rgbd:main',
            'vslamlab_droidslam_stereo = vslamlab_droidslam_stereo:main',
        ]
    },
    ext_modules=[
        CUDAExtension(
            name='droid_backends',
            include_dirs=[
                torch_include_dirs,
                osp.join(os.environ["CONDA_PREFIX"], 'include/eigen3'),
                #osp.join(os.environ["PREFIX"], 'include/eigen3')
                ],
            library_dirs=torch_library_dirs,
            sources=[
                'src/droid.cpp',
                'src/droid_kernels.cu',
                'src/correlation_kernels.cu',
                'src/altcorr_kernel.cu',
            ],
            extra_compile_args={
                'cxx': ['-O3', '-D_GLIBCXX_USE_CXX11_ABI=1'],
                'nvcc': [
                    '-O3',
                    '-D_GLIBCXX_USE_CXX11_ABI=1',
                    '-gencode=arch=compute_60,code=sm_60',
                    '-gencode=arch=compute_61,code=sm_61',
                    '-gencode=arch=compute_70,code=sm_70',
                    '-gencode=arch=compute_75,code=sm_75',
                    '-gencode=arch=compute_80,code=sm_80',
                    '-gencode=arch=compute_86,code=sm_86',
                ]
            }
        ),
    ],
    cmdclass={'build_ext': BuildExtension.with_options(no_python_abi_suffix=True)},
)
