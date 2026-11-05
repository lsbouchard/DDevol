BUILD ?= $(CURDIR)/build
RESULTS ?= $(CURDIR)/results
FIGURES ?= $(CURDIR)/figures
PYTHON ?= python3
CMAKE_OPTIONS ?=
.PHONY: build test core long finite cubic short-bounds bounded cost calculations analyze figures reproduce
build:
	sh engine/build_rev.sh "$(BUILD)" -DTPGP_ENABLE_ORBIT_EXPERIMENTS=ON $(CMAKE_OPTIONS)
test:
	ctest --test-dir "$(BUILD)" --output-on-failure
core long finite cubic short-bounds bounded cost:
	DDEVOL_BUILD="$(BUILD)" DDEVOL_RESULTS="$(RESULTS)" bash scripts/campaign.sh $@
calculations:
	DDEVOL_BUILD="$(BUILD)" DDEVOL_RESULTS="$(RESULTS)" bash scripts/campaign.sh all
analyze:
	DDEVOL_BUILD="$(BUILD)" DDEVOL_RESULTS="$(RESULTS)" bash scripts/analyze.sh
figures:
	$(PYTHON) plots/manuscript.py --results "$(RESULTS)" --output "$(FIGURES)"
reproduce:
	$(MAKE) build
	$(MAKE) calculations
	$(MAKE) analyze
	$(MAKE) figures
