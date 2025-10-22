dev-install:
	@echo "→ installing development dependencies via homebrew"
	@brew install automake autoconf libtool gettext

test:
	@echo "→ running tests"
	@uv run python -m unittest test.test_board

lint:
	@echo "→ running ruff check"
	@uv run ruff check .
	@echo "→ running mypy type check"
	@uv run mypy --explicit-package-bases autogamen/

lint-fix:
	@echo "→ running ruff check --fix"
	@uv run ruff check --fix .
	@echo "→ running mypy type check"
	@uv run mypy --explicit-package-bases autogamen/

typecheck:
	@echo "→ running mypy type check"
	@uv run mypy --explicit-package-bases autogamen/

# gnubg build targets
vendor/gnubg/configure: vendor/gnubg/autogen.sh vendor/gnubg/configure.ac
	@echo "→ running autogen.sh to generate configure script"
	cd vendor/gnubg && ./autogen.sh

vendor/gnubg/Makefile: vendor/gnubg/configure
	@echo "→ configuring gnubg (no gtk, no python, cli only)"
	cd vendor/gnubg && ./configure --without-gtk --without-python --disable-nls

vendor/gnubg/gnubg: vendor/gnubg/Makefile
	@echo "→ building gnubg"
	$(MAKE) -C vendor/gnubg

gnubg: vendor/gnubg/gnubg
	@echo "✔ gnubg built successfully at vendor/gnubg/gnubg"

gnubg-clean:
	@echo "→ cleaning gnubg build artifacts"
	@if [ -f vendor/gnubg/Makefile ]; then $(MAKE) -C vendor/gnubg clean; fi

gnubg-distclean:
	@echo "→ removing gnubg build configuration"
	@if [ -f vendor/gnubg/Makefile ]; then $(MAKE) -C vendor/gnubg distclean; fi
	@rm -f vendor/gnubg/configure vendor/gnubg/config.h.in

.PHONY: dev-install test lint lint-fix typecheck gnubg gnubg-clean gnubg-distclean
