from setuptools import find_packages, setup

package_name = 'planners'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),

    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='john',
    maintainer_email='junekyoopark@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'simple_planner = planners.simple_planner:main',
            'simple_planner_service = planners.simple_planner_service:main',
            'tompc_planner = planners.tompc_planner:main',
        ],
    },
)
