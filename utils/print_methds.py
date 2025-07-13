# %%
def print_methods(obj, show_private=False, show_dunder=False):
    """
    Print all methods and attributes of an object.

    Parameters:
        obj: The instance to introspect.
        show_private (bool): If True, include private members (starting with `_`).
        show_dunder (bool): If True, include dunder (double underscore) methods like __init__.
    """
    members = dir(obj)
    filtered = []

    for member in members:
        if member.startswith('__') and member.endswith('__'):
            if not show_dunder:
                continue
        elif member.startswith('_'):
            if not show_private:
                continue
        filtered.append(member)

    for name in filtered:
        attr = getattr(obj, name)
        if callable(attr):
            print(f"{name}()")
        else:
            print(name)

# %%
