
![image](https://github.com/user-attachments/assets/19efe967-6c67-4054-aaf7-ff5470718bd0)

Tested on DEBIAN 12
![image](https://github.com/user-attachments/assets/1a673ca6-5042-4c53-82c9-7efc790d86e7)


Here’s a comprehensive guide for setting up and running the Djavyv2 project on Debian 12, including how to address potential errors related to `matplotlib` or `mpmath`:

### 1. **Install Python and Essential Packages**

Open a terminal and switch to the root user to ensure you have the necessary permissions:

```bash
sudo su
```

Then, install Python 3, pip, and the venv module:

```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv
```

### 2. **Set Up a Virtual Environment**

Create a virtual environment for the project:

```bash
python3 -m venv env
```

Activate the virtual environment:

```bash
source env/bin/activate
```

### 3. **Clone the Project Repository**

Clone the Djavyv2 repository from GitHub:

```bash
git clone https://github.com/SecretDiscorder/Djavyv2
```

Navigate to the project directory:

```bash
cd Djavyv2
```

### 4. **Install Project Dependencies**

Install the required Python packages using pip:

```bash
pip install -r requirements.txt
```

### 5. **Refer to the Installation Guide**

For detailed installation instructions, you can refer to the `Djavy.ipynb` file provided in the repository. It may contain specific setup details or troubleshooting information.

### 6. **Run the Application**

To run the main application script, use:

```bash
python3 maindesk.py
```

### 7. **Build APK**

To build an APK for the project, use Buildozer. Ensure Buildozer is installed in your environment. If not, you can install it using pip:

```bash
pip install buildozer
```

Run the Buildozer command to build the APK:

```bash
buildozer -v android debug
```

### 8. **Troubleshooting Common Issues**

If you encounter errors related to `matplotlib` or `mpmath`, follow these steps:

1. **Check Error Messages**: Read the error messages carefully to understand what went wrong.

2. **Search for Solutions**: Use search engines to look up solutions by including specific error messages in your query. Check forums, GitHub issues, or Stack Overflow for resolutions.

3. **Install Missing Dependencies**: Sometimes, missing system libraries or packages might be the cause. Install any additional dependencies that might be required. For example:

   ```bash
   sudo apt-get install libatlas-base-dev
   ```

   This package can help resolve issues related to `matplotlib`.

4. **Update Packages**: Ensure that all Python packages are up to date:

   ```bash
   pip install --upgrade -r requirements.txt
   ```

5. **Rebuild Environment**: If problems persist, you might need to recreate the virtual environment:

   ```bash
   deactivate
   rm -rf env
   python3 -m venv env
   source env/bin/activate
   pip install -r requirements.txt
   ```

### Summary

By following these steps, you should be able to set up the Djavyv2 project, run the application, and build an APK. For any issues, especially with `matplotlib` or `mpmath`, searching online or referring to specific error messages can provide targeted solutions.
