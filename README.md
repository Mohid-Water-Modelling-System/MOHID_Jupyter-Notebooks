Jupyter Notebooks are tools for interactive computing. Here’s a step-by-step guide on how to use MOHID Jupyter Notebooks.

Step-by-Step Guide

**1. Download and Install Miniconda:**

- Go to the [Miniconda download page](https://docs.anaconda.com/miniconda/install/).
- Download the Miniconda installer for your operating system (Windows, macOS, or Linux).
- Follow the installation instructions to install Miniconda on your system.

**2. Open the Terminal or Command Prompt:**

- On Windows: Open the Anaconda Prompt or Command Prompt.
- On macOS and Linux: Open the Terminal.

**3. Create a Conda Environment:**

It's a good practice to create a separate environment for each project to avoid dependency conflicts.

- Download the file MOHID_Preprocessing_environment.yml that contains the specific packages that will be included in the environment you are going to create.
- Create a new environment named MOHID_Preprocessing_environment:

      conda env create --file MOHID_Preprocessing_environment.yml
  
**4. Activate the environment:**

    conda activate MOHID_Preprocessing_environment

**5. Install Jupyter Notebook:**

 - With the environment activated, install Jupyter Notebook using the following command:

       conda install -c conda-forge notebook

**6. Launch Jupyter Notebook:**

- Once the installation is complete, you can start Jupyter Notebook by running:

        jupyter notebook

This command will open the Jupyter Notebook interface in your default web browser.

**7. Donwload and open MOHID_Preprocessing.ipynb**
