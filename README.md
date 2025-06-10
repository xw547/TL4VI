# Targeted Learning for Variable Importance 

This repository contains the reproducible code and data for the paper **“Targeted Learning for Variable Importance” **. Currently, it hosts simulation experiments, real-data analyses, and supporting scripts. 

---

## Repository Layout

```
to_be_upload/
├── simulation/
│   ├── plug_in_n_tl/        # Plug‑in & targeted‑learning simulation scripts
│   ├── loco/                # Leave‑One‑Covariate‑Out (LOCO) experiments
│   └── bootstrap/           # Bootstrap uncertainty quantification
├── real_data/
│   ├── classification_loop.ipynb  # TL4VI on classification datasets
│   └── regression_loop_oob.ipynb  # TL4VI on regression datasets
├── scripts/
│   ├── class_eif.py         # EIF routines for classification
│   ├── eif.py               # Core efficient influence‑function code
│   ├── empirical_cdf.py     # Empirical CDF utilities
│   ├── inbag_count.py       # Bagging count functions
│   └── numerical_truth/     # Ground‑truth importance calculators
└── data/                    # Datasets for simulations and examples

README.md
requirements.txt
```

---

## Quick Start

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Simulations**

   ```bash
   python to_be_upload/simulation/plug_in_n_tl/single_simple_gam_simulation_wvar_all_e3.py
   ```

3. **Real‑data analysis**

   * Launch `to_be_upload/real_data/classification_loop.ipynb` for classification examples.
   * Launch `to_be_upload/real_data/regression_loop_oob.ipynb` for regression examples.


---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Citation

If you use this repository in your work, please cite our paper:

**Wang, X., Zhou, Y., & Hooker, G. (2024).** Targeted Learning for Variable Importance. *arXiv* preprint arXiv:2411.02221. [https://doi.org/10.48550/arXiv.2411.02221](https://doi.org/10.48550/arXiv.2411.02221)

BibTeX entry:

```bibtex
@article{wang2024targeted,
  title={Targeted Learning for Variable Importance},
  author={Wang, Xiaohan and Zhou, Yunzhe and Hooker, Giles},
  journal={arXiv preprint arXiv:2411.02221},
  year={2024},
  doi={10.48550/arXiv.2411.02221}
}
```
