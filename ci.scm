(use-modules (guix packages))
(use-modules (guix profiles))
(packages->manifest (list (primitive-load "guix.scm")))
