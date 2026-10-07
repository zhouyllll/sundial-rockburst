import inspect
import chronos
import tirex

print("== chronos Chronos2Pipeline ==")
print(inspect.signature(chronos.Chronos2Pipeline.from_pretrained))
print(inspect.signature(chronos.Chronos2Pipeline.predict))

print("== tirex load_model ==")
print(inspect.signature(tirex.load_model))
print(inspect.getdoc(tirex.load_model)[:1500] if inspect.getdoc(tirex.load_model) else None)

print("== tirex module contents ==")
import pkgutil, tirex
print([m.name for m in pkgutil.iter_modules(tirex.__path__)])
