(define-module (ci)
  #:use-module (guix packages)
  #:use-module (guix profiles))

(packages->manifest (list (primitive-load (string-append (dirname (current-filename))
                                                         "/guix.scm"))))
