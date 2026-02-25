# -*- coding: utf-8 -*-
"""
json2dict; functions for loading json files to dicts and the inverse.

Functions
---------
 - `dukit.json2dict.json_to_dict`
 - `dukit.json2dict.dict_to_json`
 - `dukit.json2dict.dict_to_json_str`
 - `dukit.json2dict.fail_float`
"""

# ============================================================================

__author__ = "Sam Scholten"
__pdoc__ = {
    "dukit.json2dict.json_to_dict": True,
    "dukit.json2dict.dict_to_json": True,
    "dukit.json2dict.dict_to_json_str": True,
    "dukit.json2dict.fail_float": True,
}

# ============================================================================

import os
import re

import numpy as np
import simplejson as json

# ============================================================================
from dukit.warn import warn

# ============================================================================


def json_to_dict(filepath: str) -> dict:
    """read the json file at filepath into a dict"""
    _, pattern = os.path.splitext(filepath)
    if pattern != ".json":
        warn("input options file did not have a json pattern/suffix")

    with open(filepath, "r", encoding="utf-8") as fip:
        jstring = _json_remove_comments(fip.read())
        dct = json.loads(jstring, object_pairs_hook=dict)
        return dct.copy()


# ============================================================================


def dict_to_json(
    dictionary: dict,
    filepath: str,
) -> None:
    """save the dict as a json in a pretty way"""

    root, pattern = os.path.splitext(filepath)
    if pattern != ".json":
        warn(f"Reformatting save filepath ({filepath}) to '.json' pattern.")
        pattern = ".json"
    path = root + pattern
    with open(path, "w", encoding="utf-8") as fip:
        fip.write(_prettyjson(dictionary))


# ============================================================================


def dict_to_json_str(obj, indent=4, maxlinelength=80):
    return _prettyjson(obj, indent, maxlinelength)


# ============================================================================


def _prettyjson(obj, indent=4, maxlinelength=80):
    """
    Renders JSON content with indentation and line splits/concatenations to
    fit maxlinelength. Only dicts, lists and basic types are supported.
    Now supports np.ndarrays.

    <Pass the dict as obj and get back a string>
    """

    items, _ = _getsubitems(obj, itemkey="", islast=True, maxlinelength=maxlinelength)
    res = _indentitems(items, indent, indentcurrent=0)
    return res


# ============================================================================


def _getsubitems(obj, itemkey: str, islast: bool, maxlinelength: int):
    items = []
    # assume we can concatenate inner content unless a child node returns an
    # expanded list
    can_concat = True

    if isinstance(obj, np.ndarray):
        obj = obj.tolist()

    isdict = isinstance(obj, dict)
    islist = isinstance(obj, list)
    istuple = isinstance(obj, tuple)

    # building json content as a list of strings or child lists
    if isdict or islist or istuple:
        if isdict:
            opening, closing, keys = ("{", "}", iter(obj.keys()))
        elif islist:
            opening, closing, keys = ("[", "]", range(0, len(obj)))
        elif istuple:
            # tuples are converted into json arrays
            opening, closing, keys = ("[", "]", range(0, len(obj)))

        if itemkey != "":
            opening = itemkey + ": " + opening
        if not islast:
            closing += ","

        # Get list of inner tokens as list
        count = 0
        subitems = []
        itemkey = ""  # not used
        for k in keys:
            count += 1
            islast_ = count == len(obj)
            itemkey_ = ""
            if isdict:
                itemkey_ = _basictype2str(k)
            # inner = (items, indent)
            inner, can_concat_ = _getsubitems(obj[k], itemkey_, islast_, maxlinelength)
            # inner can be a string or a list
            subitems.extend(inner)
            # if a child couldn't concat, then we are not able either
            can_concat = can_concat and can_concat_

        # atttempt to concat subitems if all fit within maxlinelength
        if can_concat:
            totallength = 0
            for item in subitems:
                totallength += len(item)
            totallength += len(subitems) - 1  # spaces between items
            if totallength <= maxlinelength:
                str_ = ""
                # add space between items, comma is already there
                for item in subitems:
                    str_ += item + " "
                str_ = str_.strip()
                # wrap concatenated content in a new list
                subitems = [str_]
            else:
                can_concat = False

        # attempt to concat outer brackets + inner items
        if can_concat:
            if len(opening) + totallength + len(closing) <= maxlinelength:
                items.append(opening + subitems[0] + closing)
            else:
                can_concat = False

        if not can_concat:
            items.append(opening)  # opening brackets
            # Append children to parent list as a nested list
            items.append(subitems)
            items.append(closing)  # closing brackets

    else:
        # basic types
        strobj = itemkey
        if strobj != "":
            strobj += ": "
        strobj += _basictype2str(obj)
        if not islast:
            strobj += ","
        items.append(strobj)

    return items, can_concat


# ============================================================================


def _basictype2str(obj) -> str:
    """This is a filter on objects that get sent to the json. Some types
    can't be stored literally in json files, so we can adjust for that here.
    """
    if str(obj) == "None":
        strobj = "null"
    elif isinstance(obj, type(None)):
        strobj = "null"  # seems to fail sometimes?
    elif isinstance(obj, float) and np.isnan(obj):
        strobj = '"' + str(obj) + '"'  # handle nan float objects
    elif isinstance(obj, str):
        strobj = '"' + str(obj) + '"'
    elif isinstance(obj, bool):
        strobj = {True: "true", False: "false"}[obj]
    else:
        strobj = str(obj)
    return strobj


# ============================================================================


def _indentitems(items, indent, indentcurrent):
    """Recursively traverses the list of json lines, adds indentation based
    on the current depth"""
    res = ""
    indentstr = " " * indentcurrent
    for item in items:
        if isinstance(item, (list, tuple)):
            res += _indentitems(item, indent, indentcurrent + indent)
        else:
            res += indentstr + item + "\n"
    return res


# ============================================================================


def _json_remove_comments(string, strip_space=True):
    tokenizer = re.compile(r'"|(/\*)|(\*/)|(//)|\n|\r')
    end_slashes_re = re.compile(r"(\\)*$")

    in_string = False
    in_multi = False
    in_single = False

    new_str = []
    index = 0

    for match in re.finditer(tokenizer, string):
        if not (in_multi or in_single):
            tmp = string[index : match.start()]  # noqa: E203
            if not in_string and strip_space:
                # replace white space as defined in standard
                tmp = re.sub("[ \t\n\r]+", "", tmp)
            new_str.append(tmp)
        elif not strip_space:
            # Replace comments with white space so that the JSON parser reports
            # the correct column numbers on parsing errors.
            new_str.append(" " * (match.start() - index))

        index = match.end()
        val = match.group()

        if val == '"' and not (in_multi or in_single):
            escaped = end_slashes_re.search(string, 0, match.start())

            # start of string or unescaped quote character to end string
            if not in_string or (escaped is None or len(escaped.group()) % 2 == 0):  # noqa
                in_string = not in_string
            index -= 1  # include " character in next catch
        elif not (in_string or in_multi or in_single):
            if val == "/*":
                in_multi = True
            elif val == "//":
                in_single = True
        elif val == "*/" and in_multi and not (in_string or in_single):
            in_multi = False
            if not strip_space:
                new_str.append(" " * len(val))
        elif val in "\r\n" and not (in_multi or in_string) and in_single:
            in_single = False
        elif not ((in_multi or in_single) or (val in " \r\n\t" and strip_space)):  # noqa
            new_str.append(val)

        if not strip_space:
            if val in "\r\n":
                new_str.append(val)
            elif in_multi or in_single:
                new_str.append(" " * len(val))

    new_str.append(string[index:])
    return "".join(new_str)


# ============================================================================


def fail_float(a):
    """Used in particular for reading the metadata to convert all numbers into
    floats and leave strings as strings.
    """
    try:
        return float(a)
    except ValueError:
        return a


# ============================================================================
