import setuptools
import os

FILE_PATH = os.path.dirname(os.path.realpath(__file__))   # 获取当前文件路径

with open(os.path.join(FILE_PATH, 'README.md'), 'r') as fh:
    long_description = fh.read()

requirements_path = os.path.join(FILE_PATH, 'requirements.txt')
with open(requirements_path) as f:
    required = f.read().splitlines()

setuptools.setup(
    name='climkit',  # 库名称，发布到 Pypi 就是整个项目的名称
    version='0.0.1b',   # 版本号
    author='Tingyang Song',   # 作者，发布到 Pypi 它会显示
    author_email='1057422494@qq.com',  # 作者邮箱，发布到 Pypi 它会显示
    description='Some tool function package for Climate',  # 摘要，发布到 Pypi 它会作为摘要显示
    long_description=long_description,  # 详细说明，发布到 Pypi 它会作为项目页面的说明文档显示，这里是直接从 README.md 文件读取内容传过来的
    long_description_content_type='text/markdown',  # 详细说明的渲染方式，由于是 Markdown 格式，因此设置为 markdown
    url='https://github.com/Clarmy/pydingbot',   # 项目的主页链接
    include_package_data=True,   # 项目是否包含静态数据文件
    package_data={'': ['*.csv', '*.config', '*.nl', '*.json']},   # 所包含的数据文件声明，这里的意思是包目录中所有 以.csv, .config, .nl, .json 结尾的文件在安装时都要包含，否则安装时会被忽略
    packages=setuptools.find_packages(),   # 包列表，这里使用 find_packages 函数自动扫描和识别包名，其实它是把所有包含 __init__.py 的目录作为一个包来返回的
    install_requires=required,   # 依赖包列表，这里是直接从 requirements.txt 文件中读取后传递进来的，在安装本包的时候依赖包会先置安装
    classifiers=[   # 分类，它会显示在 Pypi 项目主页左侧边栏上，可选列表：https://Pypi.org/classifiers/
        'Programming Language :: Python :: 3',
    ],
    python_requires='>=3.6'   # Python 版本限制，不满足版本限制的环境下将无法安装本包
)