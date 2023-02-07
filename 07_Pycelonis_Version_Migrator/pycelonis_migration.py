"""
Created by Álvaro Sierra (a.sierraerice@celonis.com)

Piece of code capable of migrating a code from pycelonis version 1.7.x to 2.x.

The main tool used is regular expressions, given their powerful ability to find
patterns in text. As stated in the read.me, the code isn't perfect. Its main
purpose is to save time for the end-user, not having to search and modify every
statement.

Before I start explaining thoroughly what this code actually changes, I'd like to
introduce the following notation, for the sake of generalization:
 - celonis. is the celonis object instantiated through get_celonis(*args)
 - When I write .asset it means any celonis asset such as: analyses, data pools,
 data models, action flows, data_connections ...
 - asset_name is the name of the already instantiated asset
 i.e data_pool = celonis.get_data_pools().find("pool_name)

As of now the code covers the following transformations:
General transformations
-   get_celonis(url="whatever",.... ---> get_celonis(base_url="whatever",....
-   asset_name.add_table_from_pool() ---> asset_name.add_table()

Find transformations
 -  celonis.asset.find() ---> celonis.get_assets().find()
 -  asset_name.find() ---> asset_name.get_assets().find()

Create transformations
 -  celonis.create_asset() ---> celonis.data_integration.create_asset()
 -  celonis.create_asset() ---> celonis.studio.create_asset()
 -  asset_name.create_asset() ---> asset_name.create_asset()
 -  celonis.asset ---> celonis.get_assets()
 -  asset_name.asset ---> asset_name.get_assets()
 -

TODO add the rest of patterns and the overview of classes and functions
"""

import difflib as dif
import json
from typing import Callable, Union
from pathlib import Path

import regex as re


# ==== Classes ====
class PycelonisMigrator:
    """
    Migrator Class, enables code to be migrated from pycelonis 1.7.x to pycelonis 2.x. The modify_code
    method iterates over all the search and modify patterns and firstly adds a line of code to state
    that it has been modified and afterwards, adds the get_data_frame function if needed and creates
    the diff.html file to see the difference. Finally, the modified code gets saved in a new file.

    Attributes:
            - code_path: Path to the file that you want to modify
            - pattern_list: List of tuples containing the search pattern in the first
            position and the replace pattern or function in the second position
            - file_extension: gets either .py or .ipynb
            - original code: string containing the raw read code
            - modified_code: is where the new code is going to be modified and written
            - cel_obj_names_list: list with the name of all the instantiated celonis objects
            # TODO Add methods
    """

    def __init__(self,
                 code_path: str,
                 pattern_list_: list[tuple[str, Union[str, Callable]]] = None):

        self.code_path = code_path
        self.file_extension = Path(self.code_path).suffix
        self.file_name = Path(self.code_path).stem
        self.original_code = self.read_code()
        self.modified_code = self.original_code  # Make a copy of the original
        self.json_code = self.read_code_as_json()

        self.pattern_list = pattern_list_
        self.cel_obj_names_list = self.find_celonis_object_name()

    def read_code(self) -> str:
        """Read code as a file, this function returns the raw output.

            Returns:
                -raw_code_text: Read text
        """
        with open(self.code_path) as code_text:
            raw_code_text = code_text.read()
        return raw_code_text

    def read_code_as_json(self) -> Union[dict, None]:
        """
        Function that reads raw code as a json only if the file extension is .ipynb
        If not the case it returns None.

        Returns:
            json_code: Raw notebook text read as a json, giving a dict

        """
        if self.file_extension == ".ipynb":
            json_code = json.loads(self.original_code)
            return json_code
        else:
            return None

    def replace_text(self, regex_search_pattern: str, regex_replace_pattern: str) -> None:
        """Replace regex_search_pattern with regex_replace_pattern, and store result
        in the modified code string. It firsts adds a comment to every line that is
        going to be modified

            Inputs(self):
                regex_search_pattern: pattern to search
                regex_replace_pattern: pattern to replace
        """

        pattern = re.compile(regex_search_pattern)

        self.add_comment_line_to_code(pattern=pattern,
                                      comment="CHANGED Line of code")

        self.modified_code = pattern.sub(string=self.modified_code,
                                         repl=regex_replace_pattern)

    def modify_code(self) -> None:
        """
        Method that depending on the file extension, applies regex patterns in different
        ways to update the code from 1.7.x pycelonis to 2.0.0. When archive has a .py
        extension, everything is easier, since it can be fully read as a full str and
        modified all together in one piece. On the otehr hand, notebooks need to be modified
        at cell level, since they are being raw read as if they were a JSON, so every pattern
        needs to be applied in a cell level, joining the list inside them every time to apply
        patterns on a unique chunk or string per cell.

        """
        if self.file_extension == ".py":
            # Apply replace_text function to every pattern in pattern_list (removes for loop)
            list(map(lambda x: self.replace_text(x[0], x[1]), self.pattern_list))
            self.add_get_data_frame_function_to_code()
            self.convert_base_url_kwargs()

        else:
            for cell in self.json_code["cells"]:
                if cell["cell_type"] not in ["raw", "markdown"]:
                    # Join lines in a unique string
                    self.modified_code = "#\n".join(cell["source"])
                    # Apply replace_text function to every pattern in pattern_list (removes for loop)
                    # for that concrete cell.
                    list(map(lambda x: self.replace_text(x[0], x[1]), self.pattern_list))
                    self.add_get_data_frame_function_to_code()
                    # Return modified value to the json
                    cell["source"] = self.modified_code.split("#\n")
                    self.convert_base_url_kwargs()

            self.modified_code = json.dumps(self.json_code, indent=1)


        self.compare_changes()
        self.create_new_modified_file_for_code()

    def add_comment_line_to_code(self, pattern: re.compile, comment: str) -> None:
        """This function finds the next nearest \n for every pattern match,
         and then changes it by the input comment.
            Args:
                pattern: Regex pattern used for search
                comment: Comment to be added after the regex replacement
                has taken place.
            Returns:
                None, but writes the modified code to the attribute modified
                code.
         """
        match_iterable = pattern.finditer(self.modified_code)
        # match_positions = [(match.start(), match.end()) for match in match_iterable]

        for match in match_iterable:
            if match.end() and match:
                self.modified_code = re.sub(pattern=r"\n",
                                            repl=" # " + comment + "\n",
                                            string=self.modified_code,
                                            count=1,
                                            pos=match.end())

    def convert_base_url_kwargs(self) -> None:
        """
        Function that allows the get_celonis statement to be
        changed when the arguments are kwards.
        It is used to convert the argument url within
        get_celonis to base_url when the input of the

        """
        # Find the ** name of kwards in get celonis
        pattern = re.compile(pattern=r"get_celonis\S?(\(.*?\*\*(\w*).*?\))", flags=re.MULTILINE)
        matches = pattern.finditer(string=self.modified_code)
        for match in matches:
            if match.group(2):
                pat = f"{match.group(2)}" + "\s?=\s?{[\s\S\n]+?}"
                pattern2 = re.compile(pattern=pat, flags=re.MULTILINE)
                repl_pattern = pattern2.findall(string=self.modified_code)[0] \
                    .replace('"celonis_url"', '"base_url"') \
                    .replace('"url"', '"base_url"')

                self.add_comment_line_to_code(pattern=pattern,
                                              comment="CHANGED Line of code")

                self.modified_code = pattern.sub(string=self.modified_code,
                                                 repl=repl_pattern)

    def add_get_data_frame_function_to_code(self) -> None:
        """With this function, a new function is added at the
        very beginning of the first class or function. It reads
        the new function from a plain text file and inserts it
        in the modified code.
        """

        with open("function_get_data_frame.txt") as file:
            function_text = file.read()

        if self.file_extension == ".py":
            pattern = re.compile(pattern=r"^((def|class)\s.+)", flags=re.MULTILINE)
            class_or_def = pattern.findall(string=self.modified_code)
        else:
            pattern = re.compile(pattern=r"((def|class)\s.+)", flags=re.MULTILINE)
            class_or_def = pattern.findall(string=self.original_code)

        if class_or_def:
            first_class_or_def = class_or_def[0][0]
            get_data_frame_pattern = re.compile(r"(.*?)(\S*)\s?=\s?(.*)(.get_data_frame)(\(\))")

            count_get_data_frame_statements = 0
            for match in get_data_frame_pattern.finditer(self.modified_code):
                if "#" not in match.group(1):
                    count_get_data_frame_statements += 1
                    replace_pattern = f"{match.group(1)}data_pool = {self.cel_obj_names_list[0]}" \
                                      f".data_integration.get_data_pools().find('Write here the name of the pool') " \
                                      f"# CHANGED Line of code \n" \
                                      f"{match.group(1)}data_model = data_pool.get_data_models().find" \
                                      f"('Write here the name of the data model') # CHANGED Line of code\n" \
                                      f"{match.group(1)}table = data_model.get_tables().find('Write here the name of " \
                                      f"the data model') # CHANGED Line of code\n" \
                                      f"{match.group(1)}{match.group(2)} = extract_table_from_data_pool" \
                                      f"(celonis_object=" \
                                      f"{self.cel_obj_names_list[0]}," \
                                      f"data_pool=data_pool," \
                                      f"data_model=data_model," \
                                      f"table=table)"

                    self.replace_text(get_data_frame_pattern, replace_pattern)
                    self.add_comment_line_to_code(get_data_frame_pattern, "CHECK MANUALLY, "
                                                                          "get_data_frame() no longer exists. "
                                                                          "Use the extract_table_from_data_pool "
                                                                          "function added above")

            if first_class_or_def and count_get_data_frame_statements:
                if self.file_extension == ".py":
                    pos = self.modified_code.find(first_class_or_def) - 1
                    self.modified_code = self.modified_code[:pos] + function_text + self.modified_code[pos:]
                else:
                    self.modified_code = function_text.replace("\n", "\n#\n") + "\n" + self.modified_code
                    # TODO Temporal solution, adds function at the beginning of the cell doens't care about class_or_def

    def create_new_modified_file_for_code(self) -> None:
        """
        Create a new .py or .ipynbb archive with the modified code
        """
        new_modified_file = f"{self.file_name}_migrated_automatically{self.file_extension}"
        with open(new_modified_file, "w") as migrated_file:
            migrated_file.write(self.modified_code)

    def find_celonis_object_name(self) -> list:
        """Find the name of the instantiated celonis object to add
        to the patterns
        """
        pattern = re.compile(r"(\S*)\s?=\s?get_celonis")

        global cel_obj_names_list  # TODO maybe change this, it is not best practices but
        # necessary for the convert_object_create to work
        cel_obj_names_list = [match.group(1) for match in pattern.finditer(self.original_code)]

        return cel_obj_names_list

    def compare_changes(self) -> None:
        """This function outputs an HTML file that shows the changes in both codes.
        To check the result, you need to open the outputted diff.html file in your
        web browser.
            Input:
                self
            Returns:
                None, but an HTML file is created to visualize changes in code
        """

        html_diff = dif.HtmlDiff(). \
            make_file(self.original_code.split('\n'), self.modified_code.split('\n'))

        Path('diff.html').write_text(html_diff)


# ==== Functions ====
def convert_base_url(match_obj: re.match) -> str:
    """Input function for repl regex in .sub method,
    It is used to convert the argument url within
    get_celonis to base_url

        Args:
            - match_obj: regex match object
        returns:
            - pattern: regex substitution pattern for
            the re.sub() function
    """
    if match_obj.group(4):
        celonis_obj_str = match_obj.group(1)
        get_celonis_str = match_obj.group(2)
        args_str = match_obj.group(3)
        args_str = args_str.replace(match_obj.group(4), match_obj.group(4).replace("url", "base_url"))
        repl_pattern = f"{celonis_obj_str} = {get_celonis_str}{args_str}"
        return repl_pattern


def convert_object_find(match_obj: re.match) -> str:
    """Input function for repl regex in .sub method,
    It is used to convert the (whatever) object.find to either
    .get_objects().find(object_name) or get_object()
    depending on the args taken by the find method. So if the
    celonis object is at the beginning of the statement,
    it will input the parent block (studio or data_integration)

        Args:
            - match_obj: regex match object

        returns:
            - repl_pattern: regex substitution pattern for
            the re.sub() function
    """
    cel_assets_dict = {"data_integration": ["pools", "data_models", "data_connections",
                                            "dataconnections", "jobs", "tables"],
                       "studio": ["spaces", "packages", "views", "analyses",
                                  "action_flows", "skills", "knowledge_models"]}

    cel_assets_list = cel_assets_dict["data_integration"] + cel_assets_dict["studio"]
    replace_text_part = ""
    # Create the replace text part for the regex match
    for object_ in cel_assets_list:
        if object_.replace("_", "") in match_obj.group(0) and match_obj.group(1) in cel_obj_names_list:
            cel_object_metagroup = [key for key, value in cel_assets_dict.items() if match_obj.group(3) in value][0]
            if object_ == "pools":
                replace_text_part = f"{match_obj.group(1)}.{cel_object_metagroup}.get_data_{object_}()."
            else:
                replace_text_part = f"{match_obj.group(1)}.{cel_object_metagroup}.get_{object_}()."
        elif object_.replace("_", "") in match_obj.group(0):
            if object_ == "pools":
                replace_text_part = f"{match_obj.group(1)}.get_data_{object_}()."
            else:
                replace_text_part = f"{match_obj.group(1)}.get_{object_}()."

    if match_obj.group(0):
        find_str = match_obj.group(4)
        object_name_str = match_obj.group(5)

        # FIXME: By default it adds get_analyses(), if the argument is the raw string instead of a meaningful variable
        #  name, it doesn't work

        # Decide to whether input get_objects or get_object
        if "id" in object_name_str and "analys" not in replace_text_part:
            repl_pattern = f"{replace_text_part.replace('s().', '')}{object_name_str}"
        elif "id" in object_name_str and "analys" in replace_text_part:
            repl_pattern = f"{replace_text_part.replace('ses().', 'sis')}{object_name_str}"
        elif "name" in object_name_str or "name" in object_name_str:
            repl_pattern = f"{replace_text_part}{find_str}{object_name_str}"
        else:
            repl_pattern = f"{replace_text_part}{find_str}{object_name_str}"

        return repl_pattern


def convert_object_create(match_obj: re.match) -> str:
    """Input function for repl regex in .sub method,
    It is used to convert the (whatever) create_object to either
    studio/data_integration.create_object or leave it as it is.

        Args:
            - match_obj: regex match object

        returns:
            - repl_pattern: regex substitution pattern for
            the re.sub() function
    """
    cel_assets_dict = {"data_integration": ["pool", "data_model", "data_connection",
                                            "data_job", "table"],
                       "studio": ["space", "package", "view", "analysis",
                                  "action_flow", "skill", "knowledge_model"]}

    cel_assets_list = cel_assets_dict["data_integration"] + cel_assets_dict["studio"]
    replace_text_part = ""
    # Create the replace text part for the regex match
    for object_ in cel_assets_list:
        if object_ in match_obj.group(0) and match_obj.group(1) in cel_obj_names_list:
            cel_object_metagroup = [key for key, value in cel_assets_dict.items() if match_obj.group(2) in value][0]
            replace_text_part = f"{match_obj.group(1)}.{cel_object_metagroup}" \
                                f".create_{match_obj.group(2).replace('_', '')}{match_obj.group(3)}"
        else:
            replace_text_part = f"{match_obj.group(1)}" \
                                f".create_{match_obj.group(2).replace('_', '')}{match_obj.group(3)}"

    repl_pattern = replace_text_part

    return repl_pattern


def convert_object_create_process_config(match_obj: re.match) -> str:
    """Input function for repl regex in .sub method,
    It is used to convert the data_model.create_process_config to the
    updated version. It basically changes the activity_table to
    activity_table_id, case_table to case_table_id and the rest of variables
    stay the same. On top of that, it adds the case table and activity tables
    as instantiated objects.

        Args:
            - match_obj: regex match object

        returns:
            - repl_pattern: regex substitution pattern for
            the re.sub() function
    """
    if match_obj.group(3):
        indentation = match_obj.group(1)
        datamodel_str = match_obj.group(2)
        create_proc_config_str = match_obj.group(3)
        create_proc_config_args = match_obj.group(4)
        create_proc_config_args = re.sub(string=create_proc_config_args,
                                         pattern=r"(activity_table)\s?=\s?(\S*)",
                                         repl=r"\1_id=activity_table.id")
        create_proc_config_args = re.sub(string=create_proc_config_args,
                                         pattern=r"(case_table)\s?=\s?(\S*)",
                                         repl=r"\1_id=case_table.id")
        create_proc_config_args = re.sub(string=create_proc_config_args,
                                         pattern=r"(case_column)\s?=\s?(\S*)",
                                         repl=r"case_id_column=\2")

        tables_initialized = f'''
        \n{indentation}# CHECK MANUALLY This is an exemplary initialization, it might be wrong\n
        case_table = {datamodel_str}.get_tables().find("CASE_TABLE")
        activity_table = dm.get_tables().find("ACTIVITY_TABLE")\n
        {indentation}'''

        repl_pattern = f"{tables_initialized}{datamodel_str}.{create_proc_config_str}{create_proc_config_args}"
        return repl_pattern


def convert_object_create_variable(match_obj: re.match) -> Union[str, None]:
    """Input function for repl regex in .sub method,
    It is used to convert the data_model.create_variable to the updated version.
    It basically changes the name variable for key, and variable_type for type_.

        Args:
            - match_obj: regex match object

        returns:
            - repl_pattern: regex substitution pattern for
            the re.sub() function
    """
    if match_obj.group(3):
        package_str = match_obj.group(1)
        create_variable_str = match_obj.group(2)
        create_variable_args = match_obj.group(3)
        create_variable_args = create_variable_args.replace("name=", "key=") \
            .replace("variable_type=", "type_=")
        repl_pattern = f"{package_str}.{create_variable_str}{create_variable_args}"
        return repl_pattern


def main(path: str) -> None:
    """
    This function serves as gate for the UI notebook to run the code
    of this script.

    """
    # Define the whole list of patterns
    pattern_list = find_patterns + create_patterns + general_patterns

    # Instantiate Migrator object
    Migrator = PycelonisMigrator(code_path=path,
                                 pattern_list_=pattern_list)

    Migrator.modify_code()
    print(f'The {Migrator.file_name}{Migrator.file_extension} code was successfully modified: \n'
          f'- New code file created with the following name: '
          f'{Migrator.file_name}_migrated_automatically{Migrator.file_extension}\n'
          f'- diff.html file created to easily visualize line changes by colors\n\n'
          f'If you want to check the diff.html output, run the next cell.\n'
          f'The changes will appear in an HTML file: \n'
          f'    - The file has both codes (old and new) one next to the other \n'
          f'    - The one in the left is the old with removed things highlighted in red \n'
          f'    - The one in the right is the new one with added lines in green \n'
          f'    - Feel free to scroll down and to the sides in order to visualize'
          f' both codes.\n'
          f"    - Use the buttons (f, n, t) in both codes' grey panels to quickly jump "
          f"from one modified line to other ")


# ==== Patterns ====

general_patterns = [
    (r"(\S*)\s?=\s?(get_celonis)\s?(\(((\,\s*?|)(url)\s?=).*\))", convert_base_url),
    (r"\.(add_table_from_pool)\(", r".add_table("),
]

create_patterns = [
    (r"(\S*)\.create\_(space)(\(.*\))", convert_object_create),
    (r"(\S*)\.create\_(package)(\(.*\))", convert_object_create),
    (r"(\S*)\.create\_(analysis)(\(.*\))", convert_object_create),
    (r"(\S*)\.create\_(pool)(\(.*\))", convert_object_create),
    (r"(\S*)\.create\_(datamodel)(\(.*\))", convert_object_create),
    (r"(\S*)\.create\_(data_connection)(\(.*\))", convert_object_create),
    (r"(\S*)\.create\_(table)(\(.*\))", convert_object_create),
    (r"(\S*)\.create\_(table)(\(.*\))", convert_object_create),
    (r"(\S*)\.create\_(knowledge_model)(\(.*\))", convert_object_create),
    (r"(\S*)\.create\_(view)(\(.*\))", convert_object_create),
    (r"(\S*)\.create\_(action_flow)(\(.*\))", convert_object_create),
    (r"(\S*)\.create\_(skills)(\(.*\))", convert_object_create),
    (r"(\S*)\.(create_variable)(\([\s\S\n]+?\))", convert_object_create_variable),
    (r"(\s*?)(\S*)\.(create_process_configuration)(\([\s\S\n]+?\))", convert_object_create_process_config),

]

find_patterns = [(r"(\S*)(\.(spaces)\.(find)(\(.*\)))", convert_object_find),
                 (r"(\S*)(\.(packages)\.(find)(\(.*\)))", convert_object_find),
                 (r"(\S*)(\.(analyses)\.(find)(\(.*\)))", convert_object_find),
                 (r"(\S*)(\.(pools)\.(find)(\(.*\)))", convert_object_find),
                 (r"(\S*)(\.(datamodels)\.(find)(\(.*\)))", convert_object_find),
                 (r"(\S*)(\.(data_connections)\.(find)(\(.*\)))", convert_object_find),
                 (r"(\S*)(\.(tables)\.(find)(\(.*\)))", convert_object_find),
                 (r"(\S*)(\.(knowledge_models)\.(find)(\(.*\)))", convert_object_find),
                 (r"(\S*)(\.(views)\.(find)(\(.*\)))", convert_object_find),
                 (r"(\S*)(\.(action_flows)\.(find)(\(.*\)))", convert_object_find),
                 (r"(\S*)(\.(skills)\.(find)(\(.*\)))", convert_object_find),
                 (r"(\S*)(\.(variables)\.(find)(\(.*\)))", convert_object_find),
                 (r"(\S*)\.(variables)([^.])", r"\1.get_variables()\3"),
                 (r"(\S*)\.(tables)([^.])", r"\1.get_tables()\3"),
                 (r"(\S*)\.(spaces)([^.])", r"\1.get_spaces()\3"),
                 (r"(\S*)\.(packages)([^.])", r"\1.get_packages()\3"),
                 (r"(\S*)\.(analyses)([^.])", r"\1.get_analyses()\3"),
                 (r"(\S*)\.(pools)([^.])", r"\1.get_pools()\3"),
                 (r"(\S*)\.(datamodels)([^.])", r"\1.get_data_models()\3"),
                 (r"(\S*)\.(data_connections)([^.])", r"\1.get_data_connections()\3"),
                 (r"(\S*)\.(knowledge_models)([^.])", r"\1.get_knowledge_models()\3"),
                 (r"(\S*)\.(action_flows)([^.])", r"\1.get_action_flows()\3"),
                 (r"(\S*)\.(skills)([^.])", r"\1.get_skills()\3"),
                 (r"(\S*)\.(permissions)([^.])", r"\1.team.get_permissions()\3"),
                 ]

# TODO Search for more patterns, add them. If they are recurrent create a function as with find ones
# TODO analyses.draft look it into the pycelonis_migration_removed
# TODO Recheck pycelonis changes and think of hints or import changes. For this you can use the following
#  snippet of code:
"""
import inspect

def extract_info(obj, prefix=""):
    if inspect.ismodule(obj):
        # Recursively inspect the attributes of the module
        for name, member in inspect.getmembers(obj):
            extract_info(member, prefix=prefix + name + ".")
    elif inspect.isfunction(obj) or inspect.isclass(obj):
        # Store the fully-qualified name of the function or class
        print(prefix + obj.__name__)
    else:
        # Skip attributes that are not modules, functions, or classes
        pass

# Example usage
import pandas
extract_info(pandas)
"""
