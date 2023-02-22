# Pycelonis Version Migrator
This code serves the purpose of automatically migrating a given code from pycelonis 1 to pycelonis 2
The code is not perfect (It doesn't cover every single use case), but serves as a tool to save time for
people having to migrate codes to pycelonis 2

# Project Description
## Input
The UI Script only needs the path to the location of the .py / .ipynb archive you want to migrate.

## Output
Once the code is run, it generates another .py or .ipynb file with the same name as the one you inputted but with
"_migrated_automatically" after. In every line of code that is changed there will appear a comment # CHANGED or # MANUALLY CHECK.
For more clarity on what has actually changed, the code generates a diff.html file that can be easily opened with any
browser and highlights all the changes made in either, red, green or yellow. Nevertheless, the UI script enables a
visualization of this HTML within the very UI notebook.

## Check the output once it has run
After it has run, you should check manually if there is any mistake and solve it reading the pycelonis documentation
or also reading [this article](https://celonis.github.io/pycelonis/2.0.0/tutorials/executed/04_migration/01_migration_guide/)
on how to migrate most of the biggest changes. As stated before, the code is not perfect,
but will definitely save you time.

The backend script relies on a class called PycelonisMigrator, which uses different regex patterns and mainly the
regex library to modify the text.

# Scripts
This project consists of two different scripts:

- **Pycelonis_Migration_UI.ipynb**: This is the code the final user should use. It contains a brief description of the -
overall project, and the call to the backend to perform the migration. <br /> 
- **pycelonis_migration.py**: This is the code that performs all of the operations. It is based on a class called 
PycelonisMigrator and several functions that help defining the output of the regex substitute patterns.<br />


# Additional Information
## Pycelonis version change
This project only migrates the script you provided, but bear in mind that you also need to update the python packages
so you can test the updated script outcome. It is highly encouraged for you to check which version of pycelonis are you
currently using. For this you can either:
- Type the following command in a newly opened terminal and search for the pycelonis version number: <br /> 
>pip list 
- Inside any notebook run the following piece of code:<br />
> import pycelonis <br />
> pycelonis.\__version\__

This way you can revert to the older version in case you have problems. Once you have done this, you can safely update
pycelonis to the latest version. For updating pycelonis python package follow [these guidelines.](https://celonis.github.io/pycelonis/2.0.1/tutorials/executed/01_quickstart/01_installation/)
You will need to run this command in terminal: <br />
> pip install --extra-index-url=https://pypi.celonis.cloud/ pycelonis

Note that you can select the version you want to install by adding it at the end of the command: <br />
> pip install --extra-index-url=https://pypi.celonis.cloud/ pycelonis=="2.0.1"

## Friendly reminder
- This project **does never change the original script** provided to migrate. You can safely use it since it only reads
the original, but doesn't write back on it. <br />

