# ExoLife: Predicting Exoplanet Habitability to Support Astrobiological Discovery

This project is oriented toward developing a machine learning model capable of estimating the habitability of exoplanets from astrophysical data, with the aim of supporting astrobiological research and exoplanetary exploration. The model is trained on mixed-origin data, combining observational measurements from public astronomical catalogs (such as the NASA Exoplanet Archive and the PHL Exoplanet Catalog), simulated planetary parameters, and derived metrics like equilibrium temperature, stellar flux, or Earth Similarity Index. By integrating heterogeneous sources of information, the system aims to produce a more comprehensive and generalizable habitability estimator applicable to a wide range of known and future exoplanets.

## Project Goals

- Estimate exoplanet habitability using real and simulated astrophysical features.
- Support research in astrobiology and planetary science.
- Develop a generalizable model applicable to both cataloged and hypothetical planets.
- Provide an open framework for reproducible experimentation and future integration.

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/C-HernanG/exolife.git
cd exolife
```

### 2. Create the Conda Environment

```bash
conda env create -f environment.yml
conda activate exolife
```

### 3. (Optional) Install Package in Development Mode

To install the package in development mode:

```bash
pip install -e .
```

### 4. (Optional) Register Jupyter Kernel

To use this environment inside Jupyter notebooks:

```bash
python -m ipykernel install --user --name=exolife --display-name "Python (Exolife)"
```

## Usage

### Command Line Interface

```bash
# Train a model
exolife train --config configs/baseline.yaml

# Make predictions
exolife predict --model models/best_model.pkl --data new_planets.csv

# Evaluate model
exolife evaluate --model models/best_model.pkl --test-data test_set.csv
```

## Development & Testing

### Using Make Commands

We provide a Makefile for common development tasks:

```bash
make install    # Create conda environment
make test       # Run tests
make lint       # Run code quality checks
make format     # Format code with black and isort
make clean      # Clean up cache files
make help       # Show all available commands
```

### Launch Jupyter Lab

```bash
jupyter lab
```

## Data Sources

- [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/) - Comprehensive database of confirmed exoplanets
- [PHL Exoplanet Catalog](http://phl.upr.edu/projects/habitable-exoplanets-catalog) - Habitability-focused planetary database

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- **NASA Exoplanet Archive** - For providing comprehensive exoplanet data
- **PHL Planetary Habitability Catalog** - For habitability-focused datasets
- **Astropy Community** - For astronomical computing tools
- **Astroquery Team** - For astronomical data query capabilities
- **Open Source Community** - For the amazing ML and data science tools