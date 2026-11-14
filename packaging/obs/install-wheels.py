"""Install Loki's universal wheel for the runtime selected by a distro recipe.

The existing distro wheel macros build with CPython and install for that same
interpreter. PyPy needs its own import directory and console-script shebangs,
but need not have a second copy of Flit or installer packaged for it. Run this
build-only helper with the distribution's CPython build tools; query the target
interpreter for its scheme and validate it against the wheel's Requires-Python.
The package dependency, not a runtime launcher or PATH probe, owns selection.
Staging and cleanup remain the package builder's responsibility.
"""

import argparse
import email.parser
import json
import subprocess

from installer import install
from installer.destinations import SchemeDictionaryDestination
from installer.sources import WheelFile
from packaging.specifiers import SpecifierSet


TARGET_SCHEME = """
import json
import platform
import sys
import sysconfig

paths = sysconfig.get_paths(vars={'base': sys.argv[1], 'platbase': sys.argv[1]})
paths['headers'] = paths['include']
print(json.dumps({'version': platform.python_version(), 'paths': paths}))
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python', required=True)
    parser.add_argument('--prefix', default='/usr')
    parser.add_argument('--destdir', required=True)
    parser.add_argument('wheel')
    args = parser.parse_args()
    target = json.loads(subprocess.check_output(
        [args.python, '-c', TARGET_SCHEME, args.prefix], text=True))
    with WheelFile.open(args.wheel) as source:
        wheel_metadata = email.parser.Parser().parsestr(source.read_dist_info('WHEEL'))
        if (wheel_metadata.get('Root-Is-Purelib', '').lower() != 'true'
                or wheel_metadata.get_all('Tag') != ['py3-none-any']):
            parser.error('cross-interpreter installation requires a py3-none-any wheel')
        metadata = email.parser.Parser().parsestr(source.read_dist_info('METADATA'))
        required = metadata.get('Requires-Python')
        if not required or target['version'] not in SpecifierSet(required):
            parser.error(f"{args.python} provides Python {target['version']}; "
                         f"wheel requires {required!r}")
        destination = SchemeDictionaryDestination(
            target['paths'], interpreter=args.python, script_kind='posix',
            destdir=args.destdir,
            # installer compiles using the host, not its interpreter argument.
            # Do not put CPython bytecode in a PyPy package.
            bytecode_optimization_levels=[],
        )
        install(source, destination, additional_metadata={})
    # RPM needs this for its file manifest; it cannot evaluate target sysconfig
    # before BuildRequires have been installed.
    print(target['paths']['purelib'])


if __name__ == '__main__':
    main()
