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

```bash
# Execute complete pipeline with validation
exolife dag run config/dags/dagspec.yaml
```

For a full list of available commands and usage instructions, run:

```bash
exolife --help
```

## Development & Testing

### Using Make Commands

```bash
make help           # Show all available commands
```

## Collaboration Guidelines

When running:

```bash
git commit
```

Your default editor will open a file with commented instructions. Please write your commit message at the top following the suggested format. This helps maintain a clean and consistent commit history across the project.

## License

This project is licensed under the [MIT License](LICENSE).