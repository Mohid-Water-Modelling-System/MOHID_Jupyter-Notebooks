Below is a step‐by‐step guide on how to set up and launch MOHID Jupyter Notebooks for interactive computing:

**1. Download and Install Miniconda:**

- Visit the [Miniconda download page](https://docs.anaconda.com/miniconda/install/).
- Download the Miniconda installer for your operating system (Windows, macOS, or Linux).
- Follow the installation instructions to install Miniconda on your system.

**2. Open the Terminal or Command Prompt:**

- Windows: Open the Anaconda Prompt or Command Prompt.
- macOS/Linux: Open your preferred Terminal application.


**3. Create a Conda Environment:**

It's best practice to use a dedicated environment for each project. To create the MOHID environment, follow these steps:

- Download the YAML file:
  
  Obtain the .yaml (or .yml) file that lists all required packages.

- Create the environment:
  
  Run the following command (make sure you’re in the directory where your yml file is located):


      conda env create --file ENV_NAME

  Replace ENV_NAME with the name of the environment you wish to create.

  
**4. Activate the environment:**

To work within the new environment, activate it by running:

    conda activate ENV_NAME

Replace ENV_NAME with the name of the environment you wish to activate.

**5. Launch Jupyter Notebook:**

Once the environment is activated and all necessary packages are installed, launch Jupyter Lab (or Notebook) by issuing:

     jupyter lab

This command will open the Jupyter interface in your default web browser.
Tip: If you prefer the classic Jupyter Notebook interface, use jupyter notebook instead.

**6. Open the Notebook**

Within the Jupyter interface:
- Navigate to the directory where the notebook file (.ipynb) is located.
- Click on the the notebook file (.ipynb) to open it.

By following these steps, you´ll have a fully functional MOHID Jupyter Notebook environment for interactive computing. 


