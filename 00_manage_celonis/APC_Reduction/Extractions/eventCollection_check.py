import sys
import argparse
import pandas       as pd
import os
import traceback

from pycelonis      import get_celonis
from autodocumenter import document_sql
from _utils         import parse_celonis_url

def _load_csprod_filters():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    xls = pd.ExcelFile(dir_path + r'/App Store Filters.xlsx')
    df1 = pd.read_excel(xls, 'Filters')
    df2 = pd.read_excel(xls, 'Joins')
    df3 = pd.read_excel(xls, 'Pool Variables')

    filter_dict = {}
    join_dict = {}
    para_dict = {}

    for index, row in df1.iterrows():
        if (row['Data Pool'] not in filter_dict):
            filter_dict[row['Data Pool']] = []
        filter_dict[row['Data Pool']].append(row)

    for index, row in df2.iterrows():
        if (row['Data Pool'] not in join_dict):
            join_dict[row['Data Pool']] = []
        join_dict[row['Data Pool']].append(row)

    for index, row in df3.iterrows():
        if (row['Data Pool'] not in para_dict):
            para_dict[row['Data Pool']] = []
        para_dict[row['Data Pool']].append(row)
    
    return filter_dict, join_dict, para_dict

filter_dict, join_dict, para_dict = _load_csprod_filters()

def save_requirement_sheet(sql, outputfile, pool_name, data_source):
    table_columns, table_activities, tables = _parse_sql(sql)
    return _save_table_requirements(tables, table_activities, table_columns, outputfile, pool_name, data_source)

def generate_sql(data_job):
    sql = ''
    
    for transformation in data_job.transformations.data:
        if transformation.statement is None:
            continue
        sql = sql + ' \n ' + transformation.statement
    
    return sql

def _parse_sql(sql):
    _, documentation_arr, _ = document_sql(sql, return_summary=True, on_error='raise')

    table_columns = {}
    table_activities = {}
    tables = set()
    for doc_object in documentation_arr:
        for table in doc_object['source_tables']:
            tables.add(table)
        for fullname in doc_object['source_columns']:
            if ('.' in fullname):
                table, column = fullname.split('.', 1)
                if (table not in table_columns):
                    table_columns[table] = set()
                table_columns[table].add(column)
        if (doc_object['statement_type'] == 'Create Activity'):
            for table in doc_object['source_tables']:
                if (table not in table_activities):
                    table_activities[table] = set()
                for activity in doc_object['activity_names_en']:
                    table_activities[table].add(activity)

    return (table_columns, table_activities, tables)

def _save_table_requirements(tables, table_activities, table_columns, outputfile, pool_name, data_source):
    """Saves a Requirement Sheet Excel file based on transformation script.

    :param outputfile: Path to output Excel document.
    :type outputfile: str
    """
    
    filters = dict((elm['Table Name'], elm['Filter']) for elm in filter_dict.get(pool_name, []) if elm['Data Source'] == data_source)
    delta_filters = dict((elm['Table Name'], elm['Delta Filter']) for elm in filter_dict.get(pool_name, []) if elm['Data Source'] == data_source)

    joins = {}
    for elm in join_dict.get(pool_name, []):
        if (elm['Table Name'] not in joins):
            joins[elm['Table Name']] = []
        join = dict((key, elm[key]) for key in ['parentSchema', 'parentTable', 'childTable', 'usePrimaryKeys', 'customJoinPath', 'joinFilter','order'])
        joins[elm['Table Name']].append(join)

    df_arr_tabs = []
    for table in sorted(tables):
        df_arr_tabs.append([
            table,
            filters.get(table, ''),
            delta_filters.get(table, ''),
            '\n\n'.join(str(elm['parentTable']) + ': ' + str(elm['joinFilter']) for elm in joins.get(table, []))
        ])
    df_tabs = pd.DataFrame(df_arr_tabs, columns=['Required Tables', 'Filter', 'Delta Filter', 'Joins'])

    df_arr_cols = []
    for table in sorted(table_columns.keys()):
        tab_columns = sorted(table_columns[table])
        if (len(tab_columns) == 0):
            continue
        df_arr_cols.append([table, tab_columns[0]])
        if (len(tab_columns) > 1):
            for i in range(1, len(tab_columns)):
                df_arr_cols.append(['', tab_columns[i]])
    df_cols = pd.DataFrame(df_arr_cols, columns=['Table', 'Required Columns'])

    df_arr_acts = []
    for table in sorted(table_activities.keys()):
        tab_activities = sorted(table_activities[table])
        if (len(tab_activities) == 0):
            continue
        df_arr_acts.append([table, tab_activities[0]])
        if (len(tab_activities) > 1):
            for i in range(1, len(tab_activities)):
                df_arr_acts.append(['', tab_activities[i]])
    df_act = pd.DataFrame(df_arr_acts, columns=['Table', 'Activities'])

    df_arr_paras = []
    if pool_name in para_dict:
        for parameter in para_dict[pool_name]:
            df_arr_paras.append([parameter['name'], parameter['placeholder'], parameter['values']])
    df_paras = pd.DataFrame(df_arr_paras, columns=['Parameter', 'Placeholder', 'Default Value'])

    return df_tabs, df_cols, df_act, df_paras

def check_eventCollection(url, outputdir, c):
    columns = []
    is_url = True
    try:
        url_options = parse_celonis_url(url)
    except ValueError:
        is_url = False

    if is_url:
        if (url_options['type'] == 'team'):
            for pool in c.pools:
                for job in pool.data_jobs:
                    sql = generate_sql(job)
                    if job.data_connection == 'No connection, this is a Data Job in global scope':
                        data_source = 'Global'
                    else:
                        data_source = job.data_connection.data[0].name
                    excel_name = pool.name + ' - ' + data_source + ' - Table Requirements.xlsx'
                    df1, df2, df3, df4 = save_requirement_sheet(sql, os.path.join(outputdir, excel_name), pool.name, data_source)
                    columns.append(df2)

        elif (url_options['type'] == 'pool'):
            if ('jobId' not in url_options['query']):
                raise ValueError('Specified source URL "{}" does not refer to a valid Data Job within a Data Pool.'.format(input))
            pool = c.pools.find(url_options['id'])
            data_job = pool.data_jobs.ids[url_options['query']['jobId']]
            sql = generate_sql(data_job)
            if data_job.data_connection == 'No connection, this is a Data Job in global scope':
                data_source = 'Global'
            else:
                data_source = data_job.data_connection.data[0].name
            excel_name = pool.name + ' - ' + data_source + ' - Table Requirements.xlsx'
            df1, df2, df3, df4 = save_requirement_sheet(sql, os.path.join(outputdir, excel_name), pool.name, data_source)
            columns.append(df2)
        else:
            raise ValueError('Specified URL "{}" could not be recognized.'.format(url))
        
        return columns
    else:
        with open(url, 'r', encoding="latin-1") as f:
            sql = f.read()
            data_source = url.split('-')[-2].strip()
            pool_name = data_source + ' - ' + url.split('-')[-1].strip().split('.')[0].strip()
            excel_name = pool_name + ' - Table Requirements.xlsx'
            if data_source == 'SAP ECC':
                data_source = 'SAP ERP'
            save_requirement_sheet(sql, os.path.join(outputdir, excel_name), pool_name, data_source)