# An ordinal CNN approach for the assessment of neurological damage in Parkinson's disease patients"
Code for the experimentation of the paper "An ordinal CNN approach for the assessment of neurological damage in Parkinson's disease patients" (pending review)

# Installation
Python 3.8 is required to run the experiments, along with the requirements specified in the `requirements.txt` file. To automatically install these requirements using `pip`, run the following command:

```bash
python -m pip install -r requirements.txt
```

# Usage
The `signac` project first needs to be initialized by providing the path to the folder containing:

* The `labels.csv` file with the image filenames and corresponding labels
* The `mni` directory, containing the image files in MNI152 standard space

```bash
python init.py <path to folder>
```

This will create all the validation jobs for the experimentation.

The generated jobs can then be ran using the `signac` CLI interface:

```bash
python project.py submit  # for submitting to an installed and configured scheduler
python project.py run     # for running locally
```

For more information on how to use the `signac` CLI, go to the  [`signac` documentation](https://docs.signac.io/en/latest/).

Once all validation jobs have been completed, the evaluation jobs can be created and ran:
```bash
python add_evaluation_jobs.py

# Like before, jobs can be submitted or ran locally
# Completed jobs will not be executed twice
python project.py submit # or
python project.py run
```

After all evaluation jobs are completed, the results can be extracted to spreadsheet files:
```bash
python extract_results.py
```

# Citation

### BibTeX
```bibtex
@article{BARBEROGOMEZ2021115271,
  title = {An ordinal CNN approach for the assessment of neurological damage in Parkinson’s disease patients},
  journal = {Expert Systems with Applications},
  volume = {182},
  pages = {115271},
  year = {2021},
  issn = {0957-4174},
  doi = {https://doi.org/10.1016/j.eswa.2021.115271},
  url = {https://www.sciencedirect.com/science/article/pii/S0957417421007028},
  author = {Javier Barbero-Gómez and Pedro-Antonio Gutiérrez and Víctor-Manuel Vargas and Juan-Antonio Vallejo-Casas and César Hervás-Martínez},
  keywords = {Artificial neural networks, Ordinal classification, Data augmentation, Computer-aided diagnosis}
}
```

### APA 7th edition
```
J. Barbero-Gómez, P.-A. Gutiérrez, V.-M. Vargas, J.-A. Vallejo-Casas, and C. Hervás-Martínez, “An ordinal CNN approach for the assessment of neurological damage in Parkinson’s disease patients,” Expert Systems with Applications, vol. 182, p. 115271, Nov. 2021, doi: 10.1016/j.eswa.2021.115271.
```

# Contributors
* Javier Barbero Gómez ([@javierbg](https://github.com/javierbg))
* Víctor Manuel Vargas ([@victormvy](https://github.com/victormvy))
* Pedro Antonio Gutiérrez ([@pagutierrez](https://github.com/pagutierrez))
* César Hervás-Martínez (chervas@uco.es)
