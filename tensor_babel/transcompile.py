#!/usr/bin/env python
import ast
import astunparse

from refactor import LayerCallRefactor


def refact_code(code):
    tree = ast.parse(code)

    tree_new = LayerCallRefactor().visit(tree)
    tree_new = ast.fix_missing_locations(tree_new)
    code_new = astunparse.unparse(tree_new)

    return code_new

