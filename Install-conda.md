## Installation Guide for Priism in Conda Environments

### Notes

- This installation process has been tested on Linux (Archlinux and OpenSUSE), and it should also work for other Linux distributions. However, it has **not** been tested on macOS.
- We recommend using **Miniforge** or **Miniconda** for this installation, as they are free and open-source. However, please note that access to **Anaconda’s public repository** of packages is only free for individuals and small organizations (fewer than 200 employees). A paid license is required for larger organizations or those embedding/mirroring Anaconda’s repository. For more details, refer to [Anaconda's terms of service](https://www.anaconda.com/terms-of-service).
- Using **Miniforge**, which is configured to use the **conda-forge** channel by default, is a good alternative if you want to avoid licensing concerns related to Anaconda’s repository.
- Ensure you have the appropriate permissions for copying files and modifying your environment.


### Step-by-Step Installation Instructions 

1. **Create a new conda environment:**
    ```bash
    conda create -n priism python=3.10
    ```

2. **Activate the environment:**
    ```bash
    conda activate priism
    ```

3. **Install Jupyter and necessary dependencies:**
    ```bash
    conda install conda-forge::jupyter
    conda install conda-forge::astropy
    ```
    

4. **Clone the PRIISM repository from GitHub:**
    ```bash
    git clone https://github.com/tnakazato/priism.git
    ```

5. **Navigate into the project directory and update the repository:**
    ```bash
    cd priism
    git pull
    ```

6. **Install required Python packages:**
    - This will install all necessary dependencies, including `casatasks` and `casatools` 
    ```bash
    python -m pip install -r requirements.txt
    ```

7. **Install the newest version of GCC (Optinal):**
    - If you encounter an error like "GLIBCXX_x.x.xx' not found" during the build process, install the latest version of GCC via:
    ```bash
    conda install conda-forge::gcc
    ```

8. **Manually copy the Python library (`libpython3.10.a`):**
    - Download the latest CASA package from the official website, locate the `libpython3.10.a` file (typically found in `/your_casa_folder/lib/py/lib`).
    - Copy the file to the `lib` folder of your `priism` environment (typically `/your_conda_folder/envs/priism/lib/`).

9. **Ensure that the `~/.casa/data` folder exists:**
    - If it doesn’t exist, create the folder:
    ```bash
    mkdir -p ~/.casa/data
    ```

10. **Build the project:**
    ```bash
    python setup.py build
    ```

11. **Install the project:**
    ```bash
    python setup.py install
    ```

12. **Installation Complete!**
    - The Priism package should now be installed and ready to use in the `priism` environment. You can start exploring the tutorials to get familiar with its functionalities.


---

Feel free to add any more instructions or troubleshooting tips as needed.
