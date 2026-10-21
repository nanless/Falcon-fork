import symtable
import sys
import builtins

BUILTINS = set(dir(builtins))

def undefined_globals(path):
    src = open(path).read()
    top = symtable.symtable(src, path, "exec")
    top_names = set(top.get_identifiers()) | BUILTINS
    out = []

    def walk(table, scope):
        for sym in table.get_symbols():
            name = sym.get_name()
            if sym.is_global() and not sym.is_assigned():
                if name not in top_names:
                    out.append((scope, name))
        for child in table.get_children():
            walk(child, scope + "." + child.get_name())

    walk(top, "<module>")
    return out

for path in sys.argv[1:]:
    und = undefined_globals(path)
    print(f"== {path}: {len(und)} undefined global name(s)")
    for scope, name in und:
        print(f"   {scope}: {name}")
