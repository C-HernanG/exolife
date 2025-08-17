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
make install-env
conda activate exolife
```

### 3. (Optional) Install Package in Development Mode

```bash
make install-dev
```

### 4. (Optional) Register Jupyter Kernel

```bash
make kernel
```

## ExoLife CLI Commands

The ExoLife CLI manages the unified ingestion pipeline that harmonizes multi-mission astrophysical catalogs with uncertainty propagation and cross-identification.

### Unified Ingestion Pipeline

ExoLife now uses a single, comprehensive ingestion approach that:
- Cross-identifies sources via Gaia source_id and (host, letter) pairs
- Propagates uncertainties via Monte Carlo sampling (N=1000)
- Derives features with uncertainty quantification
- Maintains data provenance and quality indicators

```bash
# Run unified ingestion pipeline
exolife merge unified_ingestion

# Execute complete pipeline with validation
exolife dag run config/dags/dagspec.yaml

# Legacy methods (all map to unified pipeline)
exolife merge baseline
exolife merge gaia_enriched
```

For a full list of available commands and usage instructions, run:

```bash
exolife --help
```

## Development & Testing

### Using Make Commands

```bash
make test           # Run tests
make lint           # Run code quality checks
make format         # Format code with black and isort
make clean          # Clean up cache files
make help           # Show all available commands
```

## Data Sources

- [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/) – Comprehensive database of confirmed exoplanets  
- [PHL Exoplanet Catalog](http://phl.upr.edu/projects/habitable-exoplanets-catalog) – Habitability-focused planetary database  
- [GAIA DR3](https://gea.esac.esa.int/archive/) – High-precision astrometric and stellar parameter data  
- [SWEET-Cat](https://www.astro.up.pt/resources/sweet-cat/) – Homogenized stellar parameters for planet-hosting stars  

## Configuration Structure

ExoLife uses a YAML-based configuration system organized in the `config/` directory:

```
config/
├── constants/          # Physical constants and parameters
│   ├── hz.yml          # Habitable zone coefficients
│   ├── feature_engineering.yml
│   ├── quality_filters.yml
│   └── drop_columns.yml
├── sources/            # Data source configurations
│   ├── nasa_exoplanet_archive_pscomppars.yml
│   ├── phl_exoplanet_catalog.yml
│   ├── gaia_dr3_astrophysical_parameters.yml
│   └── sweet_cat.yml
├── merges/             # Data merging strategies
│   ├── baseline.yml
│   ├── comprehensive.yml
│   └── gaia_enriched.yml
├── dags/               # Pipeline workflow specifications
│   └── dagspec.yaml
└── project.yml         # Project metadata
```

This modular configuration system allows for easy customization and extension of data sources, processing parameters, and workflow definitions.  

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
