# FaFeSort: A Fast and Few-shot End-to-end Neural Network for Multi-channel Spike Sorting

This repository contains the codes for our paper FaFeSort, please refer to our paper for more technical details. This work has been submitted to the IEEE for possible publication. Copyright may be transferred without notice, after which this version may no longer be accessible.

## Setup

**Clone this repository to local**
```shell
git clone git@github.com:LucasHanCEF/fafesort.git
```
**Create and activate the Conda environment**
```shell
cd fafesort
conda env create -f requirements.yml
conda activate fafesort-env
```

## Usage

We use YAML files to contain configurations, recordings, and results, which should be put in ./config, ./rec, and ./run, respectively.

The generation of various configurations can be performed by ./test/generate_yaml.py, which facilitates the validations of different pretrain-finetune pairs. This Python file generates the YAML files with different configurations deriving from ./test/target.yaml .
```shell
python ./test/generate_yaml.py
```
This command will generate both the YAML files with different configurations (stored as ./rec/target_*.yaml) and the bash script (./run.sh) for running with all these configurations.
```shell
bash ./run.sh
```
Finally, the synthesized recordings and sorting results are stored in ./config and ./run, respectively. When a configuration is provided to this program, it will automatically detect whether the current configuration has been performed before or involved recording has been generated, which will be skipped correspondingly if so.

## License
This project is licensed under the GNU GPLv3 License - see the [LICENSE](https://github.com/LucasHanCEF/fafesort/blob/main/LICENSE) file for details.