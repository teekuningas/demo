.PHONY: shell
shell:
	nix develop

.PHONY: watch
watch:
	nix develop --command nix run .#nixgl -- python main.py

.PHONY: format
format:
	black *.py
