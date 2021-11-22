import time
import os
import re
import sys
import inspect
import random
import hashlib
import logging

from urllib     import parse
from ftfy       import fix_text
from uuid       import UUID
from datetime   import datetime
from functools  import wraps

def get_user_data(c, feature):
    dat = []
    dat.append(str(datetime.now()))
    if c == "":
        dat.append("n.a.")
        dat.append("Local")
    else:
        dat.append(c._name)
        parsed_url = parse_celonis_url(c._celonis_url)
        dat.append(parsed_url["cloud_team"])
    dat.append(feature)

def truncate(s, width=30):
    if type(s) == str:
        if len(s) > width:
            s = s[: width - 3] + "..."
        return fix_text(s)
    elif type(s) == dict:
        for k in s:
            if type(s[k]) == str:
                if len(s[k]) > width:
                    s[k] = s[k][: width - 3] + "..."
        return s


def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


RETRY_ATTEMPS = 3
RETRY_DELAY = 0.1


def retry(func):
    """
  Decorator which retries a function execution a few times 
  in the case of an error, and only after having completed the retry 
  attempts (unsuccessfully) throws the exception.
  """

    @wraps(func)
    def wrapper(*args, **kwargs):
        i = 0
        exception = None
        while i <= RETRY_ATTEMPS:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                exception = e
                time.sleep(RETRY_DELAY)
                i += 1
        else:
            raise exception

    return wrapper

_LOG_DIR = os.path.join(os.environ.get("APPDATA", "./"), "ScalingProject", "logs")

def get_logger(stdout=False, to_file=False):
    if not os.path.exists(_LOG_DIR) and to_file:
        os.makedirs(_LOG_DIR)
    formatter = logging.Formatter(fmt="%(asctime)s %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    frm = inspect.stack()[1]
    filename = frm.filename.split("\\")[-1]
    mod = inspect.getmodule(frm[0])
    log_file = os.path.join(_LOG_DIR, filename + ".log")
    if to_file:
        logging.basicConfig(filename=log_file, level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    else:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    logger = logging.getLogger(mod.__name__)
    if stdout:
        screen_handler = logging.StreamHandler(stream=sys.stdout)
        screen_handler.setFormatter(formatter)
        logger.addHandler(screen_handler)
    return logger

def generate_random_hash():
    plaintext = str(random.random())
    hash_object = hashlib.md5(plaintext.encode())
    encrypted = hash_object.hexdigest()
    id = encrypted[:8] + "-" + encrypted[8:12] + "-" + encrypted[12:16] + "-" + encrypted[16:20] + "-" + encrypted[20:]
    return id

def is_hash(id):
    return id.count("-") == 4 and all(
        x in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "a", "b", "c", "d", "e", "f", "-"] for x in id
    )


def is_number(x):
    try:
        int(x)
        return True
    except ValueError:
        return False


def parse_celonis_url(url):
    address = url.split("://")[-1]
    address_arr = re.split("\/+", address)

    netloc = address_arr[0]
    path_arr = address_arr[1:]

    if netloc == "localhost:9000":
        platform = netloc
    elif netloc == "cookbook:8081":
        platform = netloc
    elif "." in netloc and netloc.split(".")[-1] == "cloud":
        platform = "cloud"
    else:
        raise ValueError('Invalid URL "{}" was specified, could not be recognized as a Celonis address.'.format(url))

    if platform == "cloud":
        server_arr = netloc.split(".")
        cloud_team = server_arr[0]
        cloud_cluster = server_arr[1]
        if "integration" in path_arr[0]:
            integration = path_arr[0]
        else:
            integration = None
    else:
        cloud_team = None
        cloud_cluster = None
        integration = None

    type = None
    id = None
    last_elm = None
    data_model_id = None
    workbench_id = None
    for elm in path_arr:
        if last_elm in ("documents", "analysis"):
            type = "analysis"
            id = elm
            break
        elif last_elm == "pools" and platform == "cloud":
            type = "pool"
            id = elm
        elif last_elm == "nav" and platform != "cloud":
            type = "folder"
            id = elm
        elif last_elm == "workbench" and platform == "cloud":
            workbench_id = elm
        elif last_elm == "process-data-models" and type == "pool":
            data_model_id = elm.split("?")[0]
            break
        elif last_elm == "data_model" and platform != "cloud":
            type = "data-model"
            id = elm
            data_model_id = elm
            break
        last_elm = elm
    if id is None:
        if platform == "cloud":
            type = "team"
            id = None
        else:
            raise ValueError('The URL "{}" could not be recognized.'.format(url))

    if type in ("analysis", "pool"):
        if platform == "cloud":
            if not is_hash(id):
                raise ValueError('Cloud address has invalid ID "' + id + '"')
        else:
            if not is_number(id):
                raise ValueError('{} address has invalid ID "'.format(platform) + id + '"')

    if "extractions" in url:
        new_url = (
            url.split("?tableId")[0].split("extractions/")[0]
            + "extractions?extractionId="
            + url.split("?tableId")[0].split("extractions/")[1]
        )
        query = dict(parse.parse_qsl(parse.urlsplit(new_url).query))
        query["jobId"] = new_url.split("data-jobs/")[1].split("/extractions")[0]
    else:
        query = dict(parse.parse_qsl(parse.urlsplit(url).query))

    return {
        "platform": platform,
        "cloud_team": cloud_team,
        "cloud_cluster": cloud_cluster,
        "integration": integration,
        "type": type,
        "id": id,
        "query": query,
        "data_model_id": data_model_id,
        "workbench_id": workbench_id,
    }


def parse_celonis_url_2(url, c):
    from urllib.parse import urlparse

    def help_find(path, attr):
        resources = path.split("/")
        for idx, r in enumerate(resources):
            if r == attr:
                return resources[idx + 1]

    parsed_url = urlparse(url)

    if not "cloud" in parsed_url.netloc:
        if "nav" in parsed_url.fragment:
            return c.folders.find(help_find(parsed_url.fragment, "nav"))
        if "documents" in parsed_url.fragment:
            return c.analyses.find(int(help_find(parsed_url.fragment, "documents")))
        if "data_model" in parsed_url.fragment:
            return c.datamodels.find(int(help_find(parsed_url.fragment, "data_model")))

    # analysis ui or event collection
    if "process-mining" in parsed_url.path:
        # we are in analysis ui
        # possible destinations:
        # - workspace
        # - analysis
        if "workspaces" in parsed_url.query:
            return c.workspaces.find(parsed_url.query.split("=")[1])
        if "analysis" in parsed_url.path:
            return c.analyses.find(parsed_url.path.split("/")[3])

    if "integration" in parsed_url.path:
        # either hybrid or normal
        if "integration-hybrid" in parsed_url.path:
            pass  # we are in hybrid scenario

        if "transformations" in parsed_url.path:
            return (
                c.pools.find(help_find(parsed_url.path, "pools"))
                .data_jobs.find(help_find(parsed_url.path, "data-jobs"))
                .transformations.find(help_find(parsed_url.path, "transformations"))
            )
        elif "data-jobs" in parsed_url.path:
            return c.pools.find(help_find(parsed_url.path, "pools")).data_jobs.find(parsed_url.query.split("=")[1].split("&")[0])
        elif "pools" in parsed_url.path:
            return c.pools.find(help_find(parsed_url.path, "pools"))

    print("Could not find the resouce you were looking for. Returning the Celonis object.")
    return c


def validate_uuid4(uuid_string):

    """
    Validate that a UUID string is in
    fact a valid uuid4.
    Happily, the uuid module does the actual
    checking for us.
    It is vital that the 'version' kwarg be passed
    to the UUID() call, otherwise any 32-character
    hex string is considered valid.
    """

    try:
        val = UUID(uuid_string, version=4)
    except ValueError:
        # If it's a value error, then the string
        # is not a valid hex code for a UUID.
        return False

    # If the uuid_string is a valid hex code,
    # but an invalid uuid4,
    # the UUID.__init__ will convert it to a
    # valid uuid4. This is bad for validation purposes.

    return val.hex == uuid_string
