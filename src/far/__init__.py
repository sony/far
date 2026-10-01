"""FAR: a latent-diffusion world model with a learned retrieval memory.

The package was called ``pmem`` before release. Run configs saved by the old name
(``_target_: pmem.generators...``) still resolve: the alias below maps every
``pmem.*`` import onto the same ``far.*`` module object, so classes are shared
rather than duplicated.
"""
import importlib
import importlib.abc
import importlib.util
import sys

from far import generators, tokenizers  # noqa: E402,F401  (public sub-packages)

_OLD, _NEW = "pmem", "far"


class _RenamedPackageFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """`import pmem.x.y` -> the already-imported `far.x.y` module, same object."""

    def find_spec(self, name, path=None, target=None):
        if name == _OLD or name.startswith(_OLD + "."):
            return importlib.util.spec_from_loader(name, self)
        return None

    def create_module(self, spec):
        return importlib.import_module(_NEW + spec.name[len(_OLD):])

    def exec_module(self, module):
        pass


if not any(isinstance(f, _RenamedPackageFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _RenamedPackageFinder())
