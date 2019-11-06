from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

print(here)

setup(name="neural",
      version="0.0.1",
      description="face neural network for game engine",
      url="https://github.com/huailiang/face-nn",
      author_email="peng_huailiang@qq.com",
      classifiers=[
          "Intended Audience :: Developers",
          "Programming Language :: Python :: 3.5",
      ],
      packages=find_packages(),
      zip_safe=False,
      install_requires=["tqdm",
                        "numpy>=1.13.3,<2.0",
                        "argparse>=1.4.0",
                        'scipy>=1.5.0',
                        'pillow',
                        'cv2',
                        "torch",
                        'torchvision',
                        'tensorboardX',
                        'scikit-image',
                        'matplotlib'],
      python_requires=">=3.5"
      )
