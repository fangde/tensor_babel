# tensor_babel
python compiler for convert tensorlayer code from 1.0 to 2.0

## to use it
cd tensor_babel/tensor_babel
python cli.py --input_file=../test/code.py --output_file==../test/compile_code.py


## checklist
1. only support python2.7
2. depends on ast, and astunparse


## how the extend

the [configure.py](tensor_babel/tensor_babel/configure.py )  file set the configuration

1. tl_layers are the layers to be refactored
2. attr_rename are the attribute name to be rename
3. args_keywords are the keywords remained to be refacor call
