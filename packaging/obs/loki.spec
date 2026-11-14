# Enable only for targets whose distro PyPy meets upstream's language floor.
# Other targets retain their existing CPython package and installation macros.
%if 0%{?fedora} >= 44
%bcond_without pypy
%{!?loki_pypy_package:%global loki_pypy_package pypy3.11}
%{!?loki_pypy_executable:%global loki_pypy_executable %{_bindir}/pypy3.11}
%else
%bcond_with pypy
%{!?loki_pypy_package:%global loki_pypy_package pypy3}
%{!?loki_pypy_executable:%global loki_pypy_executable %{_bindir}/pypy3}
%endif

%if 0%{?suse_version}
%if 0%{?suse_version} >= 1600
%define pythons %{primary_python}
%else
%define pythons python311
%endif
%global _sitelibdir %{%{pythons}_sitelib}
%else
%global _sitelibdir %{python3_sitelib}
%endif

Name:           loki
Version:        0.1.0
Release:        0
Summary:        Really really minimal-dependency coding agent
License:        MIT
URL:            https://friendly-machines.com/git/loki.git
Source0:        %{name}-%{version}.tar.gz
Source1:        install-wheels.py
BuildArch:      noarch

%if 0%{?suse_version}
BuildRequires:  %{pythons}-devel >= 3.11.10
BuildRequires:  %{pythons}-flit-core >= 3.8
BuildRequires:  %{pythons}-pip
BuildRequires:  %{pythons}-wheel
BuildRequires:  python-rpm-macros
BuildRequires:  fdupes
%if %{with pypy}
BuildRequires:  %{pythons}-installer
BuildRequires:  %{pythons}-packaging
%else
Requires:       %{pythons} >= 3.11.10
Requires:       (%{pythons} < 3.12 or %{pythons} >= 3.12.4)
%endif
%else
BuildRequires:  pyproject-rpm-macros
BuildRequires:  python3-devel >= 3.11.10
BuildRequires:  python3-flit-core >= 3.8
BuildRequires:  python3-pip
BuildRequires:  python3-wheel
%if %{with pypy}
BuildRequires:  python3-installer
BuildRequires:  python3-packaging
%else
Requires:       python3 >= 3.11.10
Requires:       (python3 < 3.12 or python3 >= 3.12.4)
%endif
%endif

%if %{with pypy}
BuildRequires:  %{loki_pypy_package}
Requires:       %{loki_pypy_package}
%endif

%description
No-nonsense coding agent with minimal dependencies.

%prep
%autosetup
# These are imported libraries, not the installed console-script entrypoints.
# Preserve their non-executable permissions rather than shipping script shebangs.
sed -i '1{/^#!/d;}' loki_agent/loki.py loki_agent/dbus_clients.py

%build
# Distribution macros use the installed backend without build isolation.
%pyproject_wheel

%install
%if %{with pypy}
sitelib=$(python3 %{SOURCE1} --python %{loki_pypy_executable} \
    --destdir %{buildroot} dist/*.whl)
printf '%%s/loki_agent/\n%%s/loki_agent-*.dist-info/\n' "$sitelib" "$sitelib" > loki.files
%else
%pyproject_install
%endif
%if 0%{?suse_version}
# CPython's pyproject_install has generated normal and optimized bytecode.
# Deduplicate the installed tree without removing either cache variant.
%fdupes %{buildroot}%{_prefix}
%endif

%check
check_imports() {
    "$1" -I -c 'import sys; sys.path.insert(0, sys.argv[1]); import loki_agent.__main__, loki_agent.acp_main, loki_agent.windows_setup, loki_agent.loki, loki_agent.dbus_clients' dist/*.whl
}
# Import from the built wheel, not the checkout, without starting a frontend,
# opening credentials, contacting a provider, or requiring a writable HOME.
%if %{with pypy}
check_imports %{loki_pypy_executable}
%else
%if 0%{?suse_version}
%python_expand check_imports $python
%else
check_imports %{__python3}
%endif
%endif

%if %{with pypy}
%files -f loki.files
%else
%files
%endif
%license COPYING
%doc README.md
%{_bindir}/loki
%{_bindir}/loki-acp
%{_bindir}/loki-setup
%if !%{with pypy}
%{_sitelibdir}/loki_agent/
%{_sitelibdir}/loki_agent-*.dist-info/
%endif

%changelog
