import sys
import sqlparse
import argparse
import re
import os
import functools

from pycelonis  import get_celonis
from _utils     import parse_celonis_url, get_logger

from os.path    import dirname, abspath
from ftfy       import fix_text

_PARENT_DIR = dirname(dirname(abspath(__file__)))
sys.path.append(_PARENT_DIR)
_DIR = dirname(abspath(__file__))
sys.path.append(_DIR)

logger = get_logger()

MAX_DISPLAYED_LIST_SIZE = 10
TOO_LONG_LIST_MESSAGE = '...\nContact App Store support for the complete list.'

JOIN_TYPES = [
    "JOIN",
    "INNER JOIN",
    "LEFT OUTER JOIN",
    "LEFT JOIN",
    "RIGHT OUTER JOIN",
    "FULL OUTER JOIN",
    "FULL JOIN",
    "NATURAL JOIN",
    "CROSS JOIN",
    "SELF JOIN",
]

with open(os.path.join(_DIR, 'sql_reserved_words.txt'), 'r') as f:
    SQL_RESERVED_WORDS = f.read().split('\n')

with open(os.path.join(_DIR, 'sql_reserved_words_column.txt'), 'r') as f:
    SQL_RESERVED_WORDS_COLUMN = f.read().split('\n')

REGEX_SQL_NEGATIVE_LOOKAHEAD = ''.join(['(?!' + word + r'(?:\Z|\s))' for word in SQL_RESERVED_WORDS]) # no match with words in list
REGEX_VARIABLES = r'''(?:\{\{\s*[a-zA-Z0-9_-]*\s*\}\})|(?:<\s*%\s*=\s*[a-zA-Z0-9_-]*\s*%\s*>)'''   # match {{any}} or <%=any%>
REGEX_TABLES = r'''(?:(?:\A|\s)FROM|(?:\A|\s)JOIN)\s+(\[?\{?\{?"?(?:[a-zA-Z0-9_]*)?\]?\}?\}?"?\.?"?\[?[a-zA-Z0-9_]*)\]?"?(?!\()'''  # match start FROM OR JOIN [any] OR {{any}}.any OR {{any}}."any" OR "any" OR "any.any" 
REGEX_COLUMNS = r'''(?:\A|\s)((?:\"?[a-zA-Z_][a-zA-Z0-9_]*\"?\.\"?[a-zA-Z_][a-zA-Z0-9_-]*\"?)|(?<!AS\s)(?<!FROM\s)"[a-zA-Z0-9_-]+")''' # "any" OR any.any OR no match in list
REGEX_TABLE_MAPPINGS = r'''(?:(?:\A|\s)FROM|(?:\A|\s)JOIN)\s+(?:\[?\{?\{?"?\w*"?\}?\}?\]?)\s+(?:AS\s+)?(?:"?\w*"?)+''' # match start FROM OR JOIN [ OR {{ OR word OR }} OR ] + AS + word
REGEX_ACTIVITY_NAMES = r'''(?:\A|\s)THEN\s*\'([^']*)\''''   # match start THEN 'any'
REGEX_FROM_SUMMARY = '(?:\A|\s)(FROM|' + '|'.join(JOIN_TYPES) + \
                     r''')\s+(?:(\(\s*\n*\s*SELECT)|(?:\{?\{?"?[a-zA-Z_][a-zA-Z0-9_]*\}?\}?"?\.)?"?([a-zA-Z_][a-zA-Z0-9_]*)\"?)''' # match start FROM OR JOIN + (SELECT OR {{any}}.any OR "any"."any" OR any
REGEX_TCURV = r'''CONVERT_CURRENCY\s*\(''' # find convert_currency function


def document_sql(sql, return_summary=False, on_error='ignore', debug=False):
    """Parses SQL script and returns corresponding docstring.

    :param sql: String representation of the SQL script to be documented.
    :type sql: str

    :param return_summary: If ``True``, a technical summary dictionary will be returned instead of just the documentation string.
    :type return_summary: bool

    :param on_error: Whether to ignore or throw exceptions. If exception is ignored, empty string will be returned as docstring. Available options are ``'ignore'`` and ``'raise'``.
    :type on_error: str

    :param debug: If ``True``, source code will be appended to the docstring for test purposes.
    :type debug: bool

    :return: Tuple with three elements. The first element is the entire documentation string, the second is a list of documentations of individual statements and the third is a dictionary containing a technical summary of the SQL script.
    :rtype: tuple
    """

    # Preprocessing SQL script for convenience
    if (sql is None):
        return ('', [], {})

    sql = standardize_sql(sql)

    # Iterating through and processing all statements in SQL scripts
    statements = sqlparse.split(sql)
    try:
        documentation_arr = list(filter(None, map(functools.partial(parse_statement, return_summary=return_summary, debug=debug), statements)))
    except Exception as e:
        if (on_error == 'raise'):
            raise e
        elif (on_error == 'ignore'):
            return ('', [], {})

    # Preparing final docstring
    if (len(documentation_arr) == 0):
        documentation = ''
    elif (len(documentation_arr) == 1):
        documentation = documentation_arr[0]['docstring'] if return_summary else documentation_arr[0]
    else:
        documentation = ''
        for i, doc_obj in enumerate(documentation_arr):
            doc_string = doc_obj['docstring'] if return_summary else doc_obj
            documentation += \
                '#' * 25 + '\n' + \
                '  S T A T E M E N T  ' + str(i + 1) + ':\n' + \
                '#' * 25 + '\n\n' + \
                doc_string + \
                '\n\n'

    # Preparing technical summary
    if (return_summary):
        documentation_summary = {
            'simple_activities': [],
            'req_tables': set(),
            'source_tables': set(),
            'created_tables': set(),
            'req_columns': {},
            'source_columns':{},
            'sorting_to_columns': {},
            'columns_to_sorting': {},
        }
        for docobj in documentation_arr:
            documentation_summary['created_tables'].add(docobj['table_name'])
            for source_table in docobj['source_tables']:
                if (source_table not in documentation_summary['created_tables']):
                    documentation_summary['source_tables'].add(source_table)
            if (docobj['statement_type'] == 'Create Activity'):
                try:
                    sorting_column = '_SORTING' if '_SORTING' in docobj['selections'] else 'SORTING'
                    sorting = int(docobj['selections'][sorting_column]) if sorting_column in docobj['selections'] else 0
                except ValueError:
                    continue
                if (sorting not in documentation_summary['sorting_to_columns']):
                    documentation_summary['sorting_to_columns'][sorting] = set()
                for colname in docobj['timestamp_columns']:
                    documentation_summary['sorting_to_columns'][sorting].add(colname)
                    documentation_summary['columns_to_sorting'][colname] = sorting
                activity_name = docobj['selections']['ACTIVITY_EN'].strip()
                if (activity_name[0] == "'" and activity_name[-1] == "'"):
                    documentation_summary['simple_activities'].append({'name': activity_name, 'sorting': sorting})
            elif (' SELECT ' in docobj['source_code']):
                selected_columns = docobj['req_columns']
                for fullname in selected_columns:
                    fullname_splitted = fullname.split('.')
                    if (len(fullname_splitted) != 2):
                        continue
                    tablename, colname = fullname_splitted
                    if (colname != '*'):
                        if (tablename not in documentation_summary['req_columns']):
                            documentation_summary['req_columns'][tablename] = set()
                        documentation_summary['req_columns'][tablename].add(colname)
                if (len(selected_columns) > 0):
                    if (docobj['table_name'] not in documentation_summary['source_columns']):
                        documentation_summary['source_columns'][docobj['table_name']] = set()
                    documentation_summary['source_columns'][docobj['table_name']].update(selected_columns)
                    documentation_summary['source_tables'].add(docobj['table_name'])

        documentation_summary['sorting_to_columns'] = [documentation_summary['sorting_to_columns'][i] for i in
                                                       sorted(documentation_summary['sorting_to_columns'].keys())]
        documentation_summary['simple_activities'] = sorted(documentation_summary['simple_activities'],
                                                            key=lambda x: x['sorting'])
    else:
        documentation_summary = {}

    return (documentation, documentation_arr, documentation_summary)

def standardize_sql(sql):
    """Standardizes an SQL script to facilitate autodocumentation of various SQL dialects.

    :param sql: String representation of the SQL script.
    :type sql: str

    :return: Standardized version of the script.
    :rtype: str
    """

    # Add spaces
    sql = sqlparse.format(sql, strip_comments=True, keyword_case='upper')
    sql = sql.replace(',', ' , ')
    sql = sql.replace('=', ' = ')
    sql = sql.replace('(', ' ( ')
    sql = sql.replace(')', ' ) ')
    sql = re.sub('\s+', ' ', sql)

    # Standardizing SQL dialects
    sql = ' ' + sql
    sql = sql.replace(' GO ', ';')
    sql = sql.replace(' SELECT DISTINCT ', ' SELECT ')
    sql = sql.replace(' CREATE COLUMN ', ' CREATE ')
    sql = sql.replace(' CREATE OR REPLACE ', ' CREATE ')
    sql = re.sub('PROMPT STATEMENT: (?:[A-Za-z0-9:]+ ?)*', '', sql)
    sql = re.sub('\s+', ' ', sql)
    sql = sql.strip()

    return sql
     
def parse_statement(s, return_summary=False, debug=False):
    """Parses SQL statement and returns corresponding docstring.

    :param s: String representation of the SQL statement
    :type s: str

    :param return_summary: If ``True``, a technical summary dictionary will be returned instead of just the documentation string.
    :type return_summary: bool

    :param debug: If ``True``, source code will be appended to the docstring for test purposes.
    :type debug: bool

    :return: Automatically generated docstring for the provided SQL statement or dictionary containing the docstring aswell as other technical data (depending on return_summary).
    :rtype: str or dict
    """

    try:
        # Check for snippet type
        s_type = get_statement_type(s)

        if (s_type not in ('Create Table', 'Create Activity', 'Create View')): # Solely Create Table/ Activity/ View left
            return {} if return_summary else ''
        
        # Get table name of the snippet 
        table_name = get_table_name(s)  # table_name = name of created table/ view or activity table for create activity 
        s_arr = s.split()

        # Extracting respective used snippets
        body_start_idx = re.search(r'(?:\A|\s)(WITH|FROM|SELECT)', s)  # search for WITH or FROM or SELECT
        header = s[:body_start_idx.start()] if body_start_idx is not None else s # header = text before WITH or FROM or SELECT
        body = s[body_start_idx.start():] if body_start_idx is not None else '' # body = text after WITH or FROM or SELECT
        body_no_strings = re.sub("'([^']*)'", '""', body) # delete special characters
        if 'FROM' in s_arr:
            from_clause = ' '.join(s_arr[s_arr.index('FROM'):])
        else: 
            from_clause = None  # From to Where clause    
        conditions = '' if 'WHERE' not in s_arr else ' '.join(s_arr[s_arr.index('WHERE') + 1:]) # Where to end clause
        joins_variables = set(re.findall(REGEX_VARIABLES, from_clause)) if from_clause is not None else [] # Find parameter in joins matching {{any}} or <%=any%>
        conditions_variables = set(re.findall(REGEX_VARIABLES, conditions)) # Find parameter in conditions/ where statement matching {{any}} or <%=any%>

        # Extracting tables used in the statement
        def get_req_tables(sql_substring):
            raw_tables = list(filter(None, sorted(re.findall(REGEX_TABLES, sql_substring))))  # extractes tables
            req_tables = []
            for t in raw_tables:
                t = t.replace('"', '').replace('[', '').replace(']', '').strip()  # delete special characters
                t = t.split('.')[-1] if '.' in t else t  # splits schema and table and returns table without schema
                if t != '' and t != 'OBJECTS' and t not in req_tables:
                    req_tables.append(t)
            return req_tables

        req_tables = get_req_tables(from_clause) if from_clause is not None else [] # tables from to Where clause
        if re.search(REGEX_TCURV, body_no_strings) is not None:
            req_tables.append('TCURV')
        source_tables = req_tables.copy()
        for table in req_tables:
            if any(x in table for x in ['TMP','TEMP','DUMMY','CEL','P2P','O2C','_CC']):
                source_tables.remove(table)
            
        # Match Table Names and Alias
        table_name_mapping_strs = re.findall(REGEX_TABLE_MAPPINGS, body_no_strings) # matches as phrases for tables
        table_name_mappings = {}

        for mapping in table_name_mapping_strs:  # get real name and alias names of tables
            if (' AS ' in mapping):
                realname, dispname = mapping.split(' AS ')
                realname = realname.split()[-1]
            else:
                try:
                    _, realname, dispname = mapping.split()
                except ValueError:
                    try:
                        _, realname = mapping.split()
                        dispname = realname
                    except ValueError: 
                        continue
            realname = realname.strip('"')
            dispname = dispname.strip('"')
            table_name_mappings[dispname] = realname # save alias-real name mapping

        # Converts shortcuts into coulmn names
        def get_real_column_names(raw_column_names):
            real_column_names = []
            for raw_name in raw_column_names:
                table_name, column_name = raw_name.split('.') if '.' in raw_name else ('', raw_name.strip('"'))
                table_name = table_name.upper().strip('"')
                if ((table_name not in source_tables and table_name not in table_name_mappings) or table_name == ''):
                    for mapping in table_name_mappings:
                        if table_name in mapping:
                            table_name = mapping
                            break
                        else:
                            continue
                else:
                    table_name = table_name_mappings[table_name] if table_name in table_name_mappings else table_name
                column_name = column_name.strip()
                full_name = '.'.join((table_name, column_name)).replace('"', '').replace('[', '').replace(']','') if table_name != '' else column_name
                if (full_name not in real_column_names and full_name not in SQL_RESERVED_WORDS):
                    real_column_names.append(full_name)
            return sorted(real_column_names)

        # Get required and source columns
        raw_column_names = re.findall(REGEX_COLUMNS, body_no_strings) # get columns
        req_columns = raw_column_names.copy()
        for column in raw_column_names:
            if any(x in column for x in SQL_RESERVED_WORDS_COLUMN):  #rather check for brakets and stuff?
                req_columns.remove(column)
        req_columns = get_real_column_names(req_columns) # get real table names for columns and delete duplicates
        source_columns = req_columns.copy() # delete TMP tables for source columns
        for column in req_columns:
            table = column.split('.')[0]
            if any(x in table for x in ['TMP','TEMP','DUMMY','CEL','P2P','O2C','_CC']):
                source_columns.remove(column)

        # Extracting FROM clause summary
        from_clause = '\n'.join((' '.join(line) for line in re.findall(REGEX_FROM_SUMMARY, s)))   # get FROM AND JOIN statements
        from_clause = re.sub(r'\(\s*\n*\s*SELECT\s*([a-zA-Z_][a-zA-Z0-9_]*)', r'SUB-SELECT \1', from_clause) # replace subselect
        from_clause = re.sub(' +', ' ', from_clause).strip() # replace + ???

        # Extracting selections (Activity Names, Timestamps)
        selection_text = ' '.join(s_arr[s_arr.index('SELECT') + 1:s_arr.index('FROM') if 'FROM' in s_arr else None]) if 'SELECT' in s_arr else s # get text from select to from/into ELSE s (all)
        selection_arr = re.split(r'''(AS\s"*[a-zA-Z0-9_.-]+"*\s*,)''', selection_text) # split whenever AS
        selection_pairs = [i + j for i, j in zip(selection_arr[::2], selection_arr[1::2])]
        selections = {}
        for pair in selection_pairs:
            keyval_arr = re.sub(r"(?:\,\s*\r?(?:\n|\Z))|\"", "", pair).strip().rsplit(" AS ", 1)[::-1]
            if len(keyval_arr) != 2:
                continue
            key, val = keyval_arr
            if key not in selections or len(val) > len(selections[key]):
                selections[key] = val
        try:
            raw_implicit_insertions = re.findall(r"\((.*)\)", header)[0].split(",")
        except IndexError:
            raw_implicit_insertions = []
        implicit_insertions = [x.strip().strip('"') for x in raw_implicit_insertions]  #delete spaces
        implicit_selections_arr = re.split(r',(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)', selection_text)
        implicit_selections = dict(zip(implicit_insertions, implicit_selections_arr))
        for key in implicit_selections:
            if key not in selections:
                selections[key] = implicit_selections[key]

        if s_type == "Create Table":
            timestamp_columns = []
            doc_str = _create_doc_string(
                description='This transformation creates a ' + ('temporary ' if 'TMP' in table_name else '') + 'table with the following name: ' + table_name,
                req_tables=req_tables,
                req_columns=req_columns,
                timestamp_columns=timestamp_columns,
                params_where=conditions_variables,
                params_join=joins_variables,
                from_clause=from_clause,
                where_clause=conditions,
                source_code=s if debug else None,
            )

        elif s_type in ("Create Activity", "Create View"):
            # Processing SQL selections (the part of the SQL statement between 'SELECT' and 'FROM')
            # and mapping them to their values, e.g.
            #  {
            #    '_CASE_KEY': 'MATERIAL._CASE_KEY',
            #    'ACTIVITY_DE': "'ERSTELLE MATERIAL'",
            #    'ACTIVITY_EN': "'CREATE MATERIAL'",
            #    ...
            #  }
            if s_type == "Create Activity":
                if "ACTIVITY_EN" not in selections:
                    selections["ACTIVITY_EN"] = "<UNDEFINED ACTIVITY>"
                if "ACTIVITY_DE" not in selections:
                    selections["ACTIVITY_DE"] = "<UNDEFINED ACTIVITY>"
                if "EVENTTIME" not in selections:
                    selections["EVENTTIME"] = "<UNDEFINED EVENTTIME>"
                if "THEN" not in selections["ACTIVITY_EN"]:
                    activity_names = re.findall(r"'(.*)'", selections["ACTIVITY_EN"])
                    activity_names = [elm for elm in activity_names if len(elm.strip("\n")) > 0]
                    activity_names_de = re.findall(r"'(.*)'", selections["ACTIVITY_DE"])
                    activity_names_de = [elm for elm in activity_names_de if len(elm.strip("\n")) > 0]
                    description = "This transformation creates an activity with the following name: " + ", ".join(
                        activity_names
                    )
                else:
                    raw_names = re.findall(REGEX_ACTIVITY_NAMES, selections["ACTIVITY_EN"])
                    activity_names = set(map(lambda x: re.sub("\s+(?:\n|\Z)", " (...)", x), raw_names))
                    raw_names_de = re.findall(REGEX_ACTIVITY_NAMES, selections["ACTIVITY_DE"])
                    activity_names_de = set(map(lambda x: re.sub("\s+(?:\n|\Z)", " (...)", x), raw_names_de))
                    description = "This transformation creates the following activities: " + ", ".join(activity_names)
                if selections["EVENTTIME"] == "<UNDEFINED EVENTTIME>":
                    timestamp_columns = []
                else:
                    timestamp_columns_raw = re.findall(REGEX_COLUMNS, selections["EVENTTIME"])
                    timestamp_columns = get_real_column_names(timestamp_columns_raw)

            else:
                view_name = s_arr[2].rsplit(".", 1)[-1]
                description = "This transformation creates a view with the following name: " + view_name
                timestamp_columns = []
            doc_str = _create_doc_string(
                description=description,
                req_tables=req_tables,
                req_columns=req_columns,
                timestamp_columns=timestamp_columns,
                params_where=conditions_variables,
                params_join=joins_variables,
                from_clause=from_clause,
                where_clause=conditions,
                source_code=s if debug else None,
            )

        if return_summary:
            ret_obj = {
                'docstring': doc_str, # transformation snippet description
                'statement_type': s_type,
                'req_tables': req_tables,
                'source_tables': source_tables,
                'req_columns': req_columns,
                'source_columns': source_columns,
                'timestamp_columns': timestamp_columns,
                'params_where': conditions_variables,
                'params_join': joins_variables,
                'from_clause': from_clause,
                'where_clause': conditions,
                'source_code': s,
                'table_name': table_name,
                'selections': selections,
                'activity_names_en': activity_names if s_type == 'Create Activity' else None,
                'activity_names_de': activity_names_de if s_type == 'Create Activity' else None
            }
            return ret_obj
        else:
            return doc_str

    except Exception as e:
        raise SQLStatementParsingError(e, s)

def get_statement_type(s):
    """Identifies the type of an SQL statement.

    :param s: An SQL statement.
    :type s_arr: str

    :return: The classification of the SQL statement, either ``'Create Activity'``, ``'Create View'``, ``'Create Table'``, or ``Other``.
    :rtype: str
    """
    s_arr = s.split()
    try:
        if (s_arr[0] == 'INSERT' and s_arr[1] == 'INTO' and 'ACTIVITIES' in s_arr[2] and 'SELECT' in s_arr and 'FROM' in s_arr):
            return 'Create Activity'
        elif (s_arr[0] == 'CREATE' and (s_arr[1] == 'VIEW' or s_arr[2] == 'VIEW')):
            return 'Create View'
        elif ((s_arr[0] == 'CREATE' and (s_arr[1] == 'TABLE' or s_arr[2] == 'TABLE')) or (
            s_arr[0] == 'SELECT' and 'INTO' in s_arr and 'FROM' in s_arr)):
            return 'Create Table'
        return 'Other'
    except IndexError:
        return 'Other'


def get_table_name(s):
    """Identifies the name of the table or view interacted with in a CREATE or INSERT statement.

    :param s: An SQL statement.
    :type s_arr: str

    :return: Name of the database table.
    :rtype: str
    """
    s_arr = s.split()
    if (s_arr[0] == 'SELECT'):
        return s_arr[s_arr.index('INTO') + 1].rsplit('.', 1)[-1].strip('"')
    else:
        return s_arr[2].rsplit('.', 1)[-1].strip('"')

def _create_doc_string(
    description, req_tables, req_columns, timestamp_columns, params_where, params_join, from_clause, where_clause,
    source_code=None
):
    """
      Returns a docstring for an SQL statement with the following format:
      "
      1. Transformation Description:
      <description>

      2. Required Tables:
      '\n'.join(<req_tables>),

      3. Required Columns:
      '\n'.join(<req_columns>)

      4. Columns used for timestamp:
      '\n'.join(<timestamp_columns>)

      5. Parameters used in where clause:
      '\n'.join(<params_where>)

      6. Parameters used in joins:
      '\n'.join(<params_join>)

      7. From clause summary:
      <from_clause>

      8. Where clause:
      <where_clause>

      Source code (for validation purposes only):
      <source_code>
        "

    Note: The source code is skipped if <source_code> is None.

    Returns
    ----------
    string
      Automatically generated docstring given the provided arguments
    """

    req_tables = _get_shortened_list(req_tables, MAX_DISPLAYED_LIST_SIZE, TOO_LONG_LIST_MESSAGE)
    req_columns = _get_shortened_list(req_columns, MAX_DISPLAYED_LIST_SIZE, TOO_LONG_LIST_MESSAGE)
    timestamp_columns = _get_shortened_list(timestamp_columns, MAX_DISPLAYED_LIST_SIZE, TOO_LONG_LIST_MESSAGE)
    params_where = _get_shortened_list(params_where, MAX_DISPLAYED_LIST_SIZE, TOO_LONG_LIST_MESSAGE)
    params_join = _get_shortened_list(params_join, MAX_DISPLAYED_LIST_SIZE, TOO_LONG_LIST_MESSAGE)

    doc_arr = [
        '1. Transformation Description:\n' + description,
        '2. Required Tables:\n' + ('\n'.join(req_tables) if len(req_tables) > 0 else 'None'),
        '3. Required Columns:\n' + ('\n'.join(req_columns) if len(req_columns) > 0 else 'None'),
        '4. Columns used for timestamp:\n' + ('\n'.join(timestamp_columns) if len(timestamp_columns) > 0 else 'None'),
        '5. Parameters used in where clause:\n' + ('\n'.join(params_where) if len(params_where) > 0 else 'None'),
        '6. Parameters used in joins:\n' + ('\n'.join(params_join) if len(params_join) > 0 else 'None'),
        # '7. From clause summary:\n'            + from_clause,
        # '8. Where clause:\n'                   + sqlparse.format(where_clause, reindent=True),
    ]
    if (source_code is not None):
        doc_arr.append('Source code (for validation purposes only):\n' + sqlparse.format(source_code, reindent=True))

    return '\n\n'.join(doc_arr)

def _get_shortened_list(arr, cutoff, message):
    if (len(arr) > cutoff):
        arr = arr[:cutoff]
        arr.append(message)
        return arr
    else:
        return arr

class SQLStatementParsingError(Exception):
    """This exception is thrown whenever an exception occured during the parsing of an SQL statement within this module."""

    def __init__(self, exception, statement):
        self.exception = exception
        self.statement = statement
        super(SQLStatementParsingError, self).__init__(statement)

    def __str__(self):
        return repr(self.exception) + '\n\nStatement:\n' + sqlparse.format(self.statement, reindent=True)

def main(url, outputfile, api_token, api_id, user_name):
    try:
        settings = parse_celonis_url(args.url)
        if settings["type"] != "pool" or "jobId" not in settings["query"]:
            raise ValueError(
                'Specified source URL "{}" does not refer to a valid Data Job within a Data Pool.'.format(args.url)
            )
        c = get_celonis(url, api_token, api_id, user_name)
        pool = c.pools.find(settings['id'])
        data_job = pool.data_jobs.ids[settings['query']['jobId']]
        sql = ''
        for transformation in data_job.transformations.data:
            if transformation.statement is None:
                continue
            sql = sql + ' \n ' + transformation.statement    
    except ValueError:
        with open(args.url, "r", encoding="latin-1") as f:
            sql = f.read()

    documentation, documentation_arr, documentation_summary = document_sql(sql, return_summary=True, on_error="raise", debug=True)

    if not os.path.exists(os.path.dirname(args.outputfile)):
        os.makedirs(args.outputfile)
    with open(args.outputfile, "w") as f:
        f.write(fix_text(documentation))
        
    logger.info('Automatic documentation saved to "{:s}".\n'.format(args.outputfile))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Script for generating documentation strings for Celonis Transformations."
    )
    parser.add_argument(
        "url",
        help="URL to a Cloud Data Job, or path to local SQL transformation script.",
    )
    parser.add_argument(
        "--outputfile",
        nargs="?",
        default="./results/docstring.txt",
        help="Path to output file where documentation string will be saved.",
    )
    parser.add_argument(
        "--api_token",
        nargs="?",
        help="Specify a valid API token for your Cloud Team or On Prem Instance.",
        default='',
    )
    parser.add_argument(
        '--api_id',
        nargs='?',
        help='Specify a valid API ID for your On Prem Instance',
        default=''
    )
    parser.add_argument(
        '--user_name',
        nargs='?',
        help='Specify your user_name for your On Prem Instance.',
        default=''
    )
    args = parser.parse_args()
    main(args.url, args.outputfile, args.api_token, args.api_id, args.user_name)