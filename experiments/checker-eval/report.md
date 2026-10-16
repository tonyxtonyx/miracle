| subset | n | TP | FP | FN | TN | **COMPLETE precision** (95% CI) | COMPLETE recall | false-COMPLETE rate | base rate resolved |
|---|---|---|---|---|---|---|---|---|---|
| online_eligible (primary) | 16 | 15 | 1 | 0 | 0 | **94%** (72%–99%) | 100% | 100% | 94% |
| all labelled | 34 | 23 | 7 | 0 | 4 | **77%** (59%–88%) | 100% | 64% | 68% |
| final, all | 29 | 19 | 6 | 0 | 4 | **76%** (57%–88%) | 100% | 60% | 66% |
| final, online-eligible | 11 | 11 | 0 | 0 | 0 | **100%** (74%–100%) | 100% | n/a | 100% |
| final, not eligible (stress test) | 18 | 8 | 6 | 0 | 4 | **57%** (33%–79%) | 100% | 60% | 44% |
| intermediate | 5 | 4 | 1 | 0 | 0 | **80%** (38%–96%) | 100% | 100% | 80% |

Per-sample decisions (decisions of repeats 0..2; C = COMPLETE, I = INCOMPLETE, ? = INVALID/ERROR):

| sample | kind | label | online-eligible | decisions | first INCOMPLETE reason |
|---|---|---|---|---|---|
| s0-dev10:astropy__astropy-8707:final | final | RESOLVED | False | CCC |  |
| s0-dev10:astropy__astropy-8707:intermediate@37 | intermediate | RESOLVED | True | CCC |  |
| s0-dev10:django__django-13821:final | final | RESOLVED | True | CCC |  |
| s0-dev10:django__django-13821:intermediate@24 | intermediate | RESOLVED | True | CCC |  |
| s0-dev10:django__django-13821:intermediate@43 | intermediate | RESOLVED | True | CCC |  |
| s0-dev10:django__django-15561:final | final | RESOLVED | True | CCC |  |
| s0-dev10:django__django-15561:intermediate@34 | intermediate | RESOLVED | True | CCC |  |
| s0-dev10:django__django-16082:final | final | RESOLVED | True | CCC |  |
| s0-dev10:django__django-17029:final | final | RESOLVED | False | CCC |  |
| s0-dev10:matplotlib__matplotlib-22865:final | final | UNRESOLVED | False | III | The diff only adds debug scripts and a reproduction script but does not contain any code changes to fix the bu |
| s0-dev10:sympy__sympy-13372:final | final | RESOLVED | False | CCC |  |
| s0-dev10:sympy__sympy-16597:final | final | UNRESOLVED | False | CCC |  |
| s0-dev10:sympy__sympy-24066:final | final | UNRESOLVED | False | CCC |  |
| s1-dev10:astropy__astropy-8707:final | final | RESOLVED | True | CCC |  |
| s1-dev10:django__django-13821:final | final | RESOLVED | True | CCC |  |
| s1-dev10:django__django-15561:final | final | RESOLVED | True | CCC |  |
| s1-dev10:django__django-16082:final | final | RESOLVED | True | CCC |  |
| s1-dev10:django__django-17029:final | final | RESOLVED | False | CCC |  |
| s1-dev10:matplotlib__matplotlib-22865:final | final | UNRESOLVED | False | CCC |  |
| s1-dev10:matplotlib__matplotlib-22865:intermediate@38 | intermediate | UNRESOLVED | True | CCC |  |
| s1-dev10:sphinx-doc__sphinx-11510:final | final | UNRESOLVED | False | III | The diff modifies the source file to emit the 'source-read' event for included files, but it does not include  |
| s1-dev10:sympy__sympy-13372:final | final | RESOLVED | True | CCC |  |
| s1-dev10:sympy__sympy-16597:final | final | UNRESOLVED | False | CCC |  |
| s1-dev10:sympy__sympy-24066:final | final | RESOLVED | False | CCC |  |
| s2-dev10:astropy__astropy-8707:final | final | RESOLVED | True | CCC |  |
| s2-dev10:django__django-13821:final | final | RESOLVED | False | CCC |  |
| s2-dev10:django__django-15561:final | final | RESOLVED | False | CCC |  |
| s2-dev10:django__django-16082:final | final | RESOLVED | True | CCC |  |
| s2-dev10:django__django-17029:final | final | RESOLVED | True | CCC |  |
| s2-dev10:matplotlib__matplotlib-22865:final | final | UNRESOLVED | False | CCC |  |
| s2-dev10:sphinx-doc__sphinx-11510:final | final | UNRESOLVED | False | CCC |  |
| s2-dev10:sympy__sympy-13372:final | final | RESOLVED | False | CCC |  |
| s2-dev10:sympy__sympy-16597:final | final | UNRESOLVED | False | III | The diff changes `oo.is_integer` to `False` but the issue specifically asks for `m.is_finite` to be `True` whe |
| s2-dev10:sympy__sympy-24066:final | final | UNRESOLVED | False | III | The diff only adds a test file to reproduce the bug but does not include any code changes to fix the underlyin |