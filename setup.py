import setuptools

setuptools.setup(
    name='tensor_babel',
    version="0.1.0",
    author='fangde liu',
    author_email='liufangde@surgicalai.cn',
    packages=['tensor_babel'],
    install_requires=['click','astunparse'],
    scripts=['bin/TensorBabel'],
    zip_safe=False
)
