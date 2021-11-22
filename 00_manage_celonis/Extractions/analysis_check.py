import os
import json
import re
import pandas   as pd

from pycelonis  import get_celonis
from _utils     import parse_celonis_url

def check_analysis(url, celonis):
    check = False
    list_of_columns = []
    workspace_url_body = url + '/process-mining/ui?workspaces='
    for workspace in celonis.workspaces:
        # print(celonis.workspaces)
        print('Getting Analysis Columns: Workspace ID - ' + workspace.data['id'])
        export_columns(celonis, workspace_url_body+workspace.data['id'], list_of_columns)

    df = pd.concat(list_of_columns)
    return df

def export_columns(c, url, list_of_columns):
    url_options = parse_celonis_url(url)

    columns_temp = []
    if url_options['type'] == 'team' and 'workspaces' in url_options['query']:
        workspace_id = url_options['query']['workspaces']
        workspace = c.workspaces.find(workspace_id)
        for analysis in workspace.analyses:
            columns_temp = columns_temp + find_columns(analysis)# columns + find_columns(analysis, columns)

    elif url_options['type'] == 'analysis':
        analysis_id = url_options['id']
        analysis = c.analyses.find(analysis_id)
        columns_temp = find_columns(analysis)

    else: 
        print('The given url is no workspace or analyses url.')

    df_col = pd.DataFrame(columns_temp, columns=['Analysis','Table', 'Required Columns']) #df_col = pd.DataFrame(columns, columns=['Table', 'Column'])
    
    list_of_columns.append(df_col)

def find_columns(analysis): #find_columns(analysis, found_columns):
    doc = analysis.published.data
    formulas = analysis.saved_formulas

    kpi_columns = []
    for kpi in formulas:
        kpi_columns = kpi_columns + re.findall(r'(\"[a-zA-Z0-9_]+\"\.\"[a-zA-Z0-9_-]+\")', kpi.data['name'])
        kpi_columns = kpi_columns +  re.findall(r'(\"[a-zA-Z0-9_]+\"\.\"[a-zA-Z0-9_-]+\")', kpi.data['template'])

    json_doc_dump = json.dumps(doc, ensure_ascii=False)
    raw_columns = re.findall(r'(\\\"[a-zA-Z0-9_]+\\\"\.\\\"[a-zA-Z0-9_-]+\\\")', json_doc_dump) + kpi_columns

    columns_temp2 = []
    for col in raw_columns:
        colum = col.replace('\\','')
        table, col = colum.split('.')
        if colum.split('.') not in columns_temp2 and '_ACTIVITIES' not in table: # and colum.split('.') not in found_columns:
            columns_temp2.append([analysis.name, table, col]) #columns.append([table, col])

    return columns_temp2
