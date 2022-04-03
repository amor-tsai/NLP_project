import os
from gensim.utils import simple_preprocess
class readTextFiles:
    def __init__(self, dirname:str = "./corpus",ext: str = "*"):
        self.dirname = dirname
        self.ext = ext

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            if self.ext != "*" and False == fname.endswith(self.ext):
                continue
            for line in open(os.path.join(self.dirname, fname), encoding='utf-8'):
                yield simple_preprocess(line)
