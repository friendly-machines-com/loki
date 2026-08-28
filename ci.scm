(use-modules (guix packages))
(packages->manifest (list (primitive-load "guix.scm")))
