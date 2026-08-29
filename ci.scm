(use-modules (guix packages))
(use-modules (guix profiles))
(packages->manifest (list (primitive-load (string-append (current-source-directory) "/guix.scm"))))
