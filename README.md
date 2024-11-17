TODO: 
1. move as much of this into a setup script as possible
2. start your own caching library based on sebastian prillo's
3. include that in the install by default, in addition to polars, seaborn, etc.


If on local and using pyenv, create a new environment with:
1. pyenv virtualenv <python-version> <env-name>
2. cd to project
3. pyenv local <env-name> # to make it activate by default in the project folder

Edit the package directory name under the src folder
Change all the fields with DEFAULT_* in the pyproject.toml

Make sure to check email to be the one you want correspondance to for this package
1. python3 -m pip install --upgrade build
2. python3 -m build
3. check whl is in the ./dist/ directory
4. pip install -e .
See [here](https://packaging.python.org/en/latest/tutorials/packaging-projects/) for more details.