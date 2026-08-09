# Habitat Loss and Fragmentation Analysis

## Overview

This repository contains Python scripts for quantifying habitat loss, habitat fragmentation, and changes in habitat fragmentation at the grid-cell scale.

## Scripts

### `Habitat_loss`

Quantifies **habitat loss at the grid-cell scale**.

This script is used to measure the occupation area of habitat area by settlement expansion within each grid cell. The resulting metrics can be used to identify spatial patterns and hotspots of habitat loss.

---

### `Habitat_frag`

Quantifies **habitat fragmentation metrics at the grid-cell scale for two time points**.

The script calculates fragmentation-related indicators separately for each time point, allowing the spatial configuration and fragmentation status of habitat to be compared through time.

---

### `Frag_change`

Quantifies **changes in habitat fragmentation at the grid-cell scale**.

This script compares the fragmentation metrics derived for the two time points and calculates the corresponding changes within each grid cell. The results can be used to identify areas where habitat fragmentation has increased or decreased.


## Outputs

The scripts generate spatially explicit, grid-level indicators describing:

- habitat loss;
- habitat fragmentation at each time point; and
- temporal changes in habitat fragmentation.

These outputs can support spatial analyses of habitat dynamics, landscape change, and biodiversity conservation.

## Requirements

The scripts are written in Python. Please ensure that the required Python packages and geospatial libraries used in each script are installed before running the analyses.

## Citation

If you use these scripts in scientific research, please cite the associated publication or repository where appropriate.

## License

Please refer to the repository license for information on usage and redistribution.