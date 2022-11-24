import pathlib

HTTP_ROOT = 'https://llandsmeer.github.io/iotf/'

def build(path: pathlib.Path, indent: int = 0):
    def emit(*a):
        print(' '*(indent*4) + ' - ' + ' '.join(map(str, a)))
    def emit_link(name, target):
        url = HTTP_ROOT + str(target).removeprefix('docs/')
        emit(f'[{name}]({url})')
    def recurse(path):
        build(path, indent+1)
    if path.is_dir():
        emit_link(path.name, path / 'index.html')
        for p in path.iterdir():
            if p.name == 'index.html':
                continue
            recurse(p)
    elif path.is_file():
        emit_link(path.name.removesuffix('.html'), path)

if __name__ == '__main__':
    build(pathlib.Path('docs'))
