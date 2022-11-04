# Asterix

Asterix is a package to simulate high-contrast imaging correction algorithms with various coronagraphs, developed at LESIA / Paris Observatory.

Please find the latest documentation (pdf file) here : https://www.dropbox.com/s/wjiwci2r6yzudvj/asterix.pdf?dl=1

Due to the continually developing nature of this package, we recommend you use the current version of the code on GitHub and keep it updated frequently.

## Installation

Clone the repository:
```bash
$ git clone https://github.com/johanmazoyer/Asterix.git
```

Create the conda environment `asterix`:
```bash
$ cd Asterix
$ conda env create --file environment.yml
```

Then install Asterix into this conda environment:
```bash
$ conda activate asterix
$ pip install -e '.'
```

## Testing and linting

Go into the package
```bash
$ cd repos/Asterix/Asterix
```

Run the pytests
```bash
$ pytest
```

or run the flake8 linter
```bash
$ flake8 . --max-line-length=121 --count --statistics
```

The test report is the displayed in your terminal in either case.

## Contributing

To contribute to Asterix, please follow the following steps:
1. Make sure your local `master` branch is up-to-date by pulling.
2. Create a new branch off `master` with a name of your choice and commit your work.
3. When done, open a PR and request a review after ensuring your branch is up-to-date with the base branch you're merging into (usually `master`),
and after running the pytests and the linter locally (see [Testing and Linting](#testing-and-linting)).
4. Iterate on the review, once it's approved it will be immediately merged.

Generale guidelines:
- Do not touch other people's branches.
- Do not touch Draft PRs.
- If you approve a PR, you can immediately merge it.