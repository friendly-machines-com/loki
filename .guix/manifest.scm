(use-modules (guix packages))
(use-modules (guix profiles))
(packages->manifest (list (primitive-load (string-append (dirname (current-filename))
                                                         "/guix.scm"))))
