# ExoLife: Predicting Exoplanet Habitability to Support Astrobiological Discovery

This project is oriented toward developing a machine learning model capable of estimating the habitability of exoplanets from astrophysical data, with the aim of supporting astrobiological research and exoplanetary exploration. The model is trained on mixed-origin data, combining observational measurements from public astronomical catalogs (such as the NASA Exoplanet Archive, PHL Exoplanet Catalog, GAIA, and SWEET-Cat), simulated planetary parameters, and derived metrics like equilibrium temperature, stellar flux, or Earth Similarity Index. By integrating heterogeneous sources of information, the system aims to produce a more comprehensive and generalizable habitability estimator applicable to a wide range of known and future exoplanets.

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

```bash
pip install -e .
```

### 4. (Optional) Register Jupyter Kernel

```bash
python -m ipykernel install --user --name=exolife --display-name "Python (Exolife)"
```

## ExoLife CLI Commands

The ExoLife CLI is used to manage data ingestion, merging strategies, and model-related workflows.  
For a full list of available commands and usage instructions, run:

```bash
exolife info
```

## Development & Testing

### Using Make Commands

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

- [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/) – Comprehensive database of confirmed exoplanets  
- [PHL Exoplanet Catalog](http://phl.upr.edu/projects/habitable-exoplanets-catalog) – Habitability-focused planetary database  
- [GAIA DR3](https://gea.esac.esa.int/archive/) – High-precision astrometric and stellar parameter data  
- [SWEET-Cat](https://www.astro.up.pt/resources/sweet-cat/) – Homogenized stellar parameters for planet-hosting stars  

## Collaboration Guidelines

When running:

```bash
git commit
```

Your default editor will open a file with commented instructions. Please write your commit message at the top following the suggested format. This helps maintain a clean and consistent commit history across the project.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- **NASA Exoplanet Archive** – For providing comprehensive exoplanet data  
- **PHL Planetary Habitability Catalog** – For habitability-focused datasets  
- **GAIA Mission** – For stellar parameters and astrometric precision  
- **SWEET-Cat Catalog** – For high-quality stellar parameters of host stars  
- **Astropy Community** – For core astronomy libraries  
- **Astroquery Team** – For astronomical data querying tools  
- **Open Source Community** – For machine learning and data science tools
