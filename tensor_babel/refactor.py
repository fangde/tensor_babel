
import ast

from configure import tl_layers as tl_layers
from configure import attr_rename as attr_rename
from configure import args_keywords as kws


def refactorFunc(node):
    print 'refactoring'
    left_node = node.args[0]
    node.args = node.args[1:-1]

    
    kwl = []
    for kw in node.keywords:
        if kw.arg in kws:
            kwl.append(kw)
            print "keywords args",kw.arg
        node.keywords.remove(kw)
        

    newNode = ast.Call(
        func=node, args=[left_node], keywords=kwl, starargs=[], kwargs=[])

    return newNode


class LayerCallRefactor(ast.NodeTransformer):
    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute):
            
            if node.func.attr in tl_layers:
                print 'ATTR', node.func.attr,(node.lineno,node.col_offset)
                return refactorFunc(node)
            self.generic_visit(node)

        if isinstance(node.func, ast.Name):
            
            if node.func.id in tl_layers:
                print 'FUNCName', node.func.id,(node.lineno,node.col_offset)
                return refactorFunc(node)
            self.generic_visit(node)

        return node

    def visit_Attribute(self, node):
        if node.attr in attr_rename.keys():
            print "rename",node.attr, (node.lineno,node.col_offset)
            node.attr = attr_rename[node.attr]

        return node
