from setuptools import setup

setup(
    name="escort_observation_bridge", version="0.1.0", packages=["escort_observation_bridge"],
    data_files=[("share/ament_index/resource_index/packages",["resource/escort_observation_bridge"]),
                ("share/escort_observation_bridge",["package.xml"])],
    install_requires=["setuptools"], zip_safe=True,
    maintainer="ECE 598 project", maintainer_email="maintainer@example.com",
    description="Causal simulation-proxy observations, coordinate checks, and trajectory logging",
    license="BSD-3-Clause",
    entry_points={"console_scripts":["actor_bridge = escort_observation_bridge.node:main"]},
)
