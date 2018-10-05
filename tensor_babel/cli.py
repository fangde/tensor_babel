import click

import ast
import astunparse

from refactor import LayerCallRefactor


@click.command()
@click.option('--input_file', help='the tl1.o code file')
@click.option('--output_file', help='the tl2.0 code file')
def transcompile(input_file, output_file):

    with open(input_file, 'r') as f:
        code = f.read()

    tree = ast.parse(code)

    tree_new = LayerCallRefactor().visit(tree)
    tree_new = ast.fix_missing_locations(tree_new)
    code_new = astunparse.unparse(tree_new)

    with open(output_file, 'w') as f:
        f.write(code_new)


if __name__ == '__main__':
    transcompile()
