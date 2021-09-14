"""
The CLI. This adaptes the Osprey CLI:
https://github.com/msmbuilder/osprey/blob/master/osprey/cli/main.py
"""
from __future__ import print_function, absolute_import, division

import sys
import argparse
import traceback

from . import receptors


def main():
    help = 'fah-prep prepares MPro structures for Folding @ Home'
    p = argparse.ArgumentParser(description=help)

    sub_parsers = p.add_subparsers(
        metavar='command',
        dest='cmd',
    )

    receptors.configure_parser(sub_parsers)  # Docs say this is correct type.

    if len(sys.argv) == 1:
        sys.argv.append('-h')

    args = p.parse_args()
    args_func(args, p)


def args_func(args, p):
    try:
        args.func(args, p)
    except RuntimeError as e:
        sys.exit("Error: %s" % e)
    except Exception as e:
        if e.__class__.__name__ not in ('ScannerError', 'ParserError'):
            message = """\
An unexpected error has occurred with FAH-prep please
consider sending the following traceback to the MSM Sense GitHub issue tracker at:
        https://github.com/RobertArbon/msm_sensitivity/issues
The error that cause this message was: 
"""
            # print(message, e, file=sys.stderr)
            print(traceback.print_tb(e.__traceback__))
            # print(message % __version__, file=sys.stderr)
