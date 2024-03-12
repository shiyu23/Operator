import sys, os, shutil, time
from distutils.core import setup 
from Cython.Build import cythonize 
from glob2 import glob
from operator import version 


start_time = time.time()
curr_dir = os.path.abspath('.')
dirs = os.listdir(curr_dir)
module_name = 'operator'
parent_path = module_name
setup_file = __file__.replace('/', '\\')
build_dir = "build"
build_tmp_dir = build_dir + "/temp"
replace_list = ['numba_func']
copy_list = ['.yml', '.yaml', '.sh', '.json']
exclude_dir = []
s = "# cython: language_level=3"


try:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel
    class bdist wheel(_bdist_wheel): 
        def finalize_options(self):
            _bdist_wheel.finalize_options(self)
            self.root is pure = False
except ImportError: 
    bdist_wheel = None 


def get_py(base_path=os.path.abspath('.'), parent_path='', name='', excepts=(), copyother=False, delC = False):
    """
    获取py文件的路径
    ：param base_path： 根路径
    ：param parent_path： 父路径
    ：param excepts： 排除文件
    ：return： py文件的迭代器
    """
    full_path = os.path.join(base_path, parent_path, name)
    init_dir = os.path.join(base_path, build_dir, parent_path, name)
    # copy init
    if os.path.exists(full_path + '/__init__.py'):
        os.makedirs(init_dir, exist_ok=True)
        shutil.copy(full_path + '/__init__.py', init_dir + '/__init__.py') 
    for filename in os.listdir(full_path):
        full_filename = os.path.join(full_path, filename)
        # copy yml
        if True in [filename.endswith(sf) for sf in copy_list]: 
            os.makedirs(init_dir, exist_ok=True)
            shutil.copy(full_path + '/' + filename, init_dir + '/' + filename)
        if os.path.isdir(full_filename) and filename != build_dir and not filename.startswith('.'):
            for f in get_py(base_path, os.path.join(parent_path, name), filename, excepts, copyother, delC): 
                yield f
        elif os.path.isfile(full_filename) and (name not in exclude_dir): 
            ext = os.path.splitext(filename)[1]
            if ext == ".c":
                if delC and os.stat(full_filename).st_mtime > start_time: 
                    os.remove (full_filename)
            elif full_filename not in excepts and os.path.splitext(filename)[1] not in ('.pyc', '.pyx'):
                if os.path.splitext(filename)[1] in ('.py', '.pyx') and not filename.startswith('__'):
                    path = os.path.join(parent_path, name, filename)
                    yield path 
        else:
            pass


def pack_pyd(): 
    # 获取py列表
    module_list = list(get_py(base_path=curr_dir, parent_path=parent_path, excepts=(setup_file,))) 
    try:
        setup(
            ext_modules=cythonize(module_list, compiler_directives={'language_level': "3"}), 
            script_args=["build_ext", "-b", build_dir,"-t", build_tmp_dir],
        )
    except Exception as ex: 
        print("error! ", str(ex)) 
    else:
        module_list = list(get_py(base_path=curr_dir, parent_path=parent_path, excepts=(setup_file,), copyother=True)) 
        
    module_list = list(get_py(base_path=curr_dir,parent_path=parent_path, excepts=(setup_file,), delC=True))
    if os.path.exists(build_tmp_dir): 
        shutil.rmtree(build_tmp_dir) 

    print("complete! time:", time.time() - start_time,'s')


def delete_c(path='.', excepts=(setup_file,)): 
    '''
    删除编译过程中生成的.c文件
    :param path:
    :param excepts: 
    :return:
    '''
    dirs = os.listdir(path) 
    for dir in dirs:
        new_dir = os.path.join(path, dir) 
        if os.path.isfile(new_dir):
            ext = os.path.splitext(new_dir)[1] 
            if ext == '.c':
                os.remove(new_dir) 
            elif os.path.isdir(new_dir): 
                delete_c(new_dir)


def replace_modules(replace_list, base_path, parent_path, build_path): 
    src_path = os.path.join(base_path, parent_path)
    dist_path = os.path.join(base_path, build_path, parent_path) 
    suffix = 'pyd'
    if os.name == 'posix': 
        suffix= 'so'
    for mod in replace_list: 
        src_file = glob(os.path.join(src_path,'./%s*.%s' % (mod, suffix))) 
        py_ver = ''.join(sys.version.split('.')[:2])
        src_file = [i for i in src_file if py_ver in i][0]
        dist_file = glob(os.path.join(dist_path,'./%s*.%s'%(mod, suffix)))[0] 
        shutil.copy(src_file,dist_file)


def build_package(): 
    # build bdist_wheel 
    import os
    from setuptools import setup, find_packages 
    
    os.chdir(build_dir)
    modules = find_packages(where='.') 
    setup(
        name=module_name, 
        packages=modules,
        package_data={mod: ['*.so', '*.pyd'] +['*' + i for i in copy_list] for mod in modules}, python_requires='>=3.5',
        version=version.version,
        cmdclass={'bdist_wheel': bdist_wheel}, 
    )


if __name__ == '__main__': 
    try: 
        shutil.rmtree(build_dir, ignore_errors=True) 
        pack_pyd()
        replace_modules(replace_list, os.path.abspath('.'), parent_path, build_dir) 
        build_package() 
    except Exception as e: 
            print(str(e)) 
    finally: 
            delete_c() 
