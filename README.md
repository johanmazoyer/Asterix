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