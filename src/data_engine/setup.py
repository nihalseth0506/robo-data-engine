from setuptools import find_packages, setup

package_name = 'data_engine'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Nihal Sanjay Seth',
    maintainer_email='seth.nihal.work@gmail.com',
    description='Universal robot data collection and synchronization pipeline',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'hello_node = data_engine.hello_node:main',
            'ur5e_node = data_engine.ur5e_node:main',
            'sync_node = data_engine.sync_node:main',
            'teleop_node = data_engine.teleop_node:main',
        ],
    },
)