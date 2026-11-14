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
URL:            https://github.com/friendly-machines-com/loki
Source0:        %{name}-%{version}.tar.gz
BuildArch:      noarch

%if 0%{?suse_version}
BuildRequires:  %{pythons}-devel >= 3.11.10
BuildRequires:  %{pythons}-flit-core >= 3.8
BuildRequires:  %{pythons}-pip
BuildRequires:  %{pythons}-wheel
BuildRequires:  python-rpm-macros
Requires:       %{pythons} >= 3.11.10
Requires:       (%{pythons} < 3.12 or %{pythons} >= 3.12.4)
%else
BuildRequires:  pyproject-rpm-macros
BuildRequires:  python3-devel >= 3.11.10
BuildRequires:  python3-flit-core >= 3.8
BuildRequires:  python3-pip
BuildRequires:  python3-wheel
Requires:       python3 >= 3.11.10
Requires:       (python3 < 3.12 or python3 >= 3.12.4)
%endif

%description
No-nonsense coding agent with minimal dependencies.

%prep
%autosetup

%build
# Distribution macros use the installed backend without build isolation.
%pyproject_wheel

%install
%pyproject_install

%files
%license COPYING
%doc README.md
%{_bindir}/loki
%{_bindir}/loki-acp
%{_bindir}/loki-setup
%{_sitelibdir}/loki_agent/
%{_sitelibdir}/loki_agent-*.dist-info/
