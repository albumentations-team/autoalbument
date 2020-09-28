## Installation

1. Create a new virtual environment.
2. Install Albumentations from the `updated_transforms` branch: `pip install -e git+https://github.com/albumentations-team/albumentations@updated_transforms#egg=albumentations`
3. Clone this repository: `git clone git@github.com:creafz/autoalbument.git`
4. Install AutoAlbument: `pip install -e autoalbument/`
5. Create files for a new experiment: `autoalbument create /path/to/directory`
6. Follow the instructions from `autoalbument create`.
7. Found policies will be saved in `<working directory>/policy`. Load the policy with Albumentations: `transform = A.load</path/to/policy.json)`
