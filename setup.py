from setuptools import find_packages, setup

# Keep the upstream entry points and include the HRC extension packages.
setup(packages=find_packages(include=[
    "lw_benchhub", "lw_benchhub.*", "lw_benchhub_tasks", "lw_benchhub_tasks.*",
    "lw_benchhub_rl", "lw_benchhub_rl.*", "policy", "policy.*",
    "hrc_bench", "hrc_bench.*", "isaac_human", "isaac_human.*", "tools", "tools.*",
]))
