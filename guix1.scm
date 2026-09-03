;;; Guix package definition for Loki.
;;; Build with:   guix build -f guix.scm
;;; Install with: guix package -f guix.scm

(use-modules (guix packages)
             (guix git-download) 
             (guix build-system pyproject)
             (guix build utils)
             (guix gexp)
             ((guix licenses) #:prefix license:)
             (gnu packages python-build))

(define %source-directory
  (dirname (current-filename)))

(define %git-file?
  (git-predicate %source-directory))

(package
  (name "loki")
  (version "0.1.0")
  (source (local-file %source-directory
                      "loki-checkout"
                       #:recursive? #t 
                       #:select?
                       (lambda (file stat)
                         (and (or (not %git-file?)
                                  (%git-file? file stat))
                              (not (string-suffix? ".scm" file))))))
  (build-system pyproject-build-system)
  (arguments
   (list #:tests? #f ; unittest suite wants credentials/network; run separately
         #:phases
         #~(modify-phases %standard-phases
             ;; flit_core only ships the Python package; the desktop entry is
             ;; installed from the source tree.
             (add-after 'install 'install-desktop-file
               (lambda* (#:key outputs #:allow-other-keys)
                 (let ((apps (string-append #$output "/share/applications")))
                   (mkdir-p apps)
                   (copy-file "loki.desktop"
                              (string-append apps "/loki.desktop"))))))))
  (native-inputs (list python-flit-core))
  (home-page #f)
  (synopsis "Really really minimal-dependency coding agent")
  (description
   "Loki is a coding agent for ECMA-48 consoles with a minimal dependency
footprint.  It speaks the Anthropic and OpenAI protocols and offers shell,
file-editing, search, web and subagent tools.  Run it inside your own VM or
container for isolation.")
  (license license:expat))
