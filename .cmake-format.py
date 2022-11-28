# ----------------------------------
# Options affecting listfile parsing
# ----------------------------------
with section("parse"):

    # Specify structure for custom cmake functions
    additional_commands = {
        "cc_binary": {
            "flags": ["ADD_RUNTARGET"],
            "kwargs": {
                "DEPS": "*",
                "INC": {
                    "kwargs": {"INTERFACE": "*", "PRIVATE": "*", "PUBLIC": "*"},
                    "pargs": 0,
                },
                "LIBDIRS": {
                    "kwargs": {"INTERFACE": "*", "PRIVATE": "*", "PUBLIC": "*"},
                    "pargs": "*",
                },
                "PKGDEPS": "*",
                "PROPERTIES": {"kwargs": {"EXPORT_NAME": 1, "OUTPUT_NAME": 1}},
                "SRCS": "*",
            },
            "pargs": "1+",
        },
        "cc_library": {
            "flags": ["STATIC", "SHARED"],
            "kwargs": {
                "DEPS": {
                    "kwargs": {"INTERFACE": "*", "PRIVATE": "*", "PUBLIC": "*"},
                    "pargs": "*",
                },
                "INC": {
                    "kwargs": {"INTERFACE": "*", "PRIVATE": "*", "PUBLIC": "*"},
                    "pargs": 0,
                },
                "LIBDIRS": {
                    "kwargs": {"INTERFACE": "*", "PRIVATE": "*", "PUBLIC": "*"},
                    "pargs": "*",
                },
                "PKGDEPS": "*",
                "PROPERTIES": {
                    "kwargs": {
                        "ARCHIVE_OUTPUT_NAME": 1,
                        "EXPORT_NAME": 1,
                        "INTERFACE_INCLUDE_DIRECTORIES": 1,
                        "LIBRARY_OUTPUT_NAME": 1,
                        "OUTPUT_NAME": 1,
                        "SOVERSION": 1,
                        "SUFFIX": 1,
                        "VERSION": 1,
                    }
                },
                "SRCS": "*",
            },
            "pargs": "1+",
        },
        "cc_test": {
            "kwargs": {
                "ARGV": "*",
                "DEPS": "*",
                "LABELS": "*",
                "PKGDEPS": "*",
                "SRCS": "*",
                "TEST_DEPS": "*",
                "WORKING_DIRECTORY": "*",
            },
            "pargs": 1,
        },
        "check_call": {
            "flags": [
                "OUTPUT_QUIET",
                "ERROR_QUIET",
                "OUTPUT_STRIP_TRAILING_WHITESPACE",
                "ERROR_STRIP_TRAILING_WHITESPACE",
            ],
            "kwargs": {
                "COMMAND": "*",
                "ENCODING": "1",
                "ERROR_FILE": "1",
                "ERROR_VARIABLE": "1",
                "INPUT_FILE": "1",
                "OUTPUT_FILE": "1",
                "OUTPUT_VARIABLE": "1",
                "RESULTS_VARIABLE": "1",
                "RESULT_VARIABLE": "1",
                "TIMEOUT": "1",
                "WORKING_DIRECTORY": "1",
            },
        },
        "check_pyoneline": {
            "kwargs": {"ERROR_VARIABLE": 1, "OUTPUT_VARIABLE": 1},
            "pargs": "+",
        },
        "create_debian_binary_packages": {
            "kwargs": {"DEPS": "*", "OUTPUTS": "*"},
            "pargs": [3, "+"],
        },
        "create_debian_depsrepo": {"pargs": [3, "+"]},
        "create_debian_packages": {
            "kwargs": {"DEPS": "*", "OUTPUTS": "*"},
            "pargs": [{"flags": ["FORCE_PBUILDER"], "nargs": "+"}],
        },
        "debhelp": {"pargs": ["1+"], "spelling": "DEBHELP"},
        "exportvars": {
            "kwargs": {"VARS": "+"},
            "pargs": "1+",
            "spelling": "EXPORTVARS",
        },
        "format_and_lint": {
            "kwargs": {"CC": "*", "CMAKE": "*", "JS": "*", "PY": "*", "SHELL": "*"}
        },
        "get_debs": {"pargs": [3, "*"]},
        "gresource": {"kwargs": {"DEPENDS": "+", "SRCDIR": 1}, "pargs": 2},
        "gtk_doc_add_module": {
            "kwargs": {
                "FIXREFOPTS": "*",
                "IGNOREHEADERS": "*",
                "LIBRARIES": "*",
                "LIBRARY_DIRS": "*",
                "SOURCE": "*",
                "SUFFIXES": "*",
                "XML": 1,
            },
            "pargs": 1,
        },
        "importvars": {
            "kwargs": {"VARS": "+"},
            "pargs": "1+",
            "spelling": "IMPORTVARS",
        },
        "join": {"kwargs": {"GLUE": 1}, "pargs": [1, "+"]},
        "pkg_find": {"kwargs": {"PKG": "*"}},
        "stage_files": {
            "kwargs": {"FILES": "*", "LIST": 1, "SOURCEDIR": 1, "STAGE": 1}
        },
        "tangent_addtest": {
            "kwargs": {
                "COMMAND": "+",
                "CONFIGURATIONS": "+",
                "DEPENDS": "+",
                "LABELS": "+",
                "NAME": 1,
                "WORKING_DIRECTORY": 1,
            }
        },
        "tangent_extract_svg": {"kwargs": {"EXPORT": 1, "OUTPUT": 1, "SRC": 1}},
        "tangent_fetchobj": {"kwargs": {"OUTDIR": 1}, "pargs": 2},
        "tangent_rmark_render": {
            "kwargs": {"DEPENDS": 1, "FORMAT": 1, "OUTPUT": 1, "PAGENO": 1, "UUID": 1},
            "pargs": 1,
        },
        "tangent_unzip": {
            "kwargs": {"OUTPUT": "1+", "WORKING_DIRECTORY": 1},
            "pargs": "1+",
        },
        "travis_decrypt": {"kwargs": {}, "pargs": [3]},
    }

    # Override configurations per-command where available
    override_spec = {}

    # Specify variable tags.
    vartags = []

    # Specify property tags.
    proptags = []

# -----------------------------
# Options affecting formatting.
# -----------------------------
with section("format"):

    # Disable formatting entirely, making cmake-format a no-op
    disable = False

    # How wide to allow formatted cmake files
    line_width = 100

    # How many spaces to tab for indent
    tab_size = 2

    # If true, lines are indented using tab characters (utf-8 0x09) instead of
    # <tab_size> space characters (utf-8 0x20). In cases where the layout would
    # require a fractional tab character, the behavior of the  fractional
    # indentation is governed by <fractional_tab_policy>
    use_tabchars = False

    # If <use_tabchars> is True, then the value of this variable indicates how
    # fractional indentions are handled during whitespace replacement. If set to
    # 'use-space', fractional indentation is left as spaces (utf-8 0x20). If set
    # to `round-up` fractional indentation is replaced with a single tab character
    # (utf-8 0x09) effectively shifting the column to the next tabstop
    fractional_tab_policy = "use-space"

    # If an argument group contains more than this many sub-groups (parg or kwarg
    # groups) then force it to a vertical layout.
    max_subgroups_hwrap = 3

    # If a positional argument group contains more than this many arguments, then
    # force it to a vertical layout.
    max_pargs_hwrap = 6

    # If a cmdline positional group consumes more than this many lines without
    # nesting, then invalidate the layout (and nest)
    max_rows_cmdline = 3

    # If true, separate flow control names from their parentheses with a space
    separate_ctrl_name_with_space = False

    # If true, separate function names from parentheses with a space
    separate_fn_name_with_space = False

    # If a statement is wrapped to more than one line, than dangle the closing
    # parenthesis on its own line.
    dangle_parens = False

    # If the trailing parenthesis must be 'dangled' on its on line, then align it
    # to this reference: `prefix`: the start of the statement,  `prefix-indent`:
    # the start of the statement, plus one indentation  level, `child`: align to
    # the column of the arguments
    dangle_align = "prefix"

    # If the statement spelling length (including space and parenthesis) is
    # smaller than this amount, then force reject nested layouts.
    min_prefix_chars = 4

    # If the statement spelling length (including space and parenthesis) is larger
    # than the tab width by more than this amount, then force reject un-nested
    # layouts.
    max_prefix_chars = 10

    # If a candidate layout is wrapped horizontally but it exceeds this many
    # lines, then reject the layout.
    max_lines_hwrap = 2

    # What style line endings to use in the output.
    line_ending = "unix"

    # Format command names consistently as 'lower' or 'upper' case
    command_case = "canonical"

    # Format keywords consistently as 'lower' or 'upper' case
    keyword_case = "unchanged"

    # A list of command names which should always be wrapped
    always_wrap = []

    # If true, the argument lists which are known to be sortable will be sorted
    # lexicographicall
    enable_sort = True

    # If true, the parsers may infer whether or not an argument list is sortable
    # (without annotation).
    autosort = False

    # By default, if cmake-format cannot successfully fit everything into the
    # desired linewidth it will apply the last, most agressive attempt that it
    # made. If this flag is True, however, cmake-format will print error, exit
    # with non-zero status code, and write-out nothing
    require_valid_layout = False

    # A dictionary mapping layout nodes to a list of wrap decisions. See the
    # documentation for more information.
    layout_passes = {}

# ------------------------------------------------
# Options affecting comment reflow and formatting.
# ------------------------------------------------
with section("markup"):

    # What character to use for bulleted lists
    bullet_char = "*"

    # What character to use as punctuation after numerals in an enumerated list
    enum_char = "."

    # If comment markup is enabled, don't reflow the first comment block in each
    # listfile. Use this to preserve formatting of your copyright/license
    # statements.
    first_comment_is_literal = False

    # If comment markup is enabled, don't reflow any comment block which matches
    # this (regex) pattern. Default is `None` (disabled).
    literal_comment_pattern = None

    # Regular expression to match preformat fences in comments default=
    # ``r'^\s*([`~]{3}[`~]*)(.*)$'``
    fence_pattern = "^\\s*([`~]{3}[`~]*)(.*)$"

    # Regular expression to match rulers in comments default=
    # ``r'^\s*[^\w\s]{3}.*[^\w\s]{3}$'``
    ruler_pattern = "^\\s*[^\\w\\s]{3}.*[^\\w\\s]{3}$"

    # If a comment line matches starts with this pattern then it is explicitly a
    # trailing comment for the preceeding argument. Default is '#<'
    explicit_trailing_pattern = "#<"

    # If a comment line starts with at least this many consecutive hash
    # characters, then don't lstrip() them off. This allows for lazy hash rulers
    # where the first hash char is not separated by space
    hashruler_min_length = 10

    # If true, then insert a space between the first hash char and remaining hash
    # chars in a hash ruler, and normalize its length to fill the column
    canonicalize_hashrulers = True

    # enable comment markup parsing and reflow
    enable_markup = False

# ----------------------------
# Options affecting the linter
# ----------------------------
with section("lint"):

    # a list of lint codes to disable
    disabled_codes = ["C0113"]

    # regular expression pattern describing valid function names
    function_pattern = "[0-9a-z_]+"

    # regular expression pattern describing valid macro names
    macro_pattern = "[0-9A-Z_]+"

    # regular expression pattern describing valid names for variables with global
    # (cache) scope
    global_var_pattern = "[A-Z][0-9A-Z_]+"

    # regular expression pattern describing valid names for variables with global
    # scope (but internal semantic)
    internal_var_pattern = "_[A-Z][0-9A-Z_]+"

    # regular expression pattern describing valid names for variables with local
    # scope
    local_var_pattern = "[a-z][a-z0-9_]+"

    # regular expression pattern describing valid names for privatedirectory
    # variables
    private_var_pattern = "_[0-9a-z_]+"

    # regular expression pattern describing valid names for public directory
    # variables
    public_var_pattern = "[A-Z][0-9A-Z_]+"

    # regular expression pattern describing valid names for function/macro
    # arguments and loop variables.
    argument_var_pattern = "[a-z][a-z0-9_]+"

    # regular expression pattern describing valid names for keywords used in
    # functions or macros
    keyword_pattern = "[A-Z][0-9A-Z_]+"

    # In the heuristic for C0201, how many conditionals to match within a loop in
    # before considering the loop a parser.
    max_conditionals_custom_parser = 2

    # Require at least this many newlines between statements
    min_statement_spacing = 1

    # Require no more than this many newlines between statements
    max_statement_spacing = 2
    max_returns = 6
    max_branches = 12
    max_arguments = 5
    max_localvars = 15
    max_statements = 50

# -------------------------------
# Options affecting file encoding
# -------------------------------
with section("encode"):

    # If true, emit the unicode byte-order mark (BOM) at the start of the file
    emit_byteorder_mark = False

    # Specify the encoding of the input file. Defaults to utf-8
    input_encoding = "utf-8"

    # Specify the encoding of the output file. Defaults to utf-8. Note that cmake
    # only claims to support utf-8 so be careful when using anything else
    output_encoding = "utf-8"

# -------------------------------------
# Miscellaneous configurations options.
# -------------------------------------
with section("misc"):

    # A dictionary containing any per-command configuration overrides. Currently
    # only `command_case` is supported.
    per_command = {}
