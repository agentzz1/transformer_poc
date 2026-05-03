# ============================================================================
# Makefile for Transformer Encoder VHDL PoC (GHDL)
# ============================================================================
# Usage:
#   make          - compile and run simulation
#   make elab     - analyse + elaborate only
#   make sim      - run simulation (requires prior elaboration)
#   make wave     - open waveform in GTKWave
#   make clean    - remove build artefacts
# ============================================================================

GHDL      := ghdl
GTKWAVE   := gtkwave
WORKDIR   := work
STD       := --std=08

# VHDL source files in dependency order
SRC := \
	clog2_pkg.vhd \
	gemm_os.vhd \
	gemm_os_adapter.vhd \
	softmax.vhd \
	scalar_ops.vhd \
	layernorm.vhd \
	weight_mem.vhd \
	gemm_mm.vhd \
	mha_controller.vhd \
	ffn.vhd \
	residual_add.vhd \
	control_unit.vhd \
	encoder_block.vhd \
	tb_encoder_block.vhd

TB_TOP    := tb_encoder_block
WAVE_FILE := wave.ghw

.PHONY: all elab sim wave clean

all: clean elab sim

$(WORKDIR):
	mkdir -p $(WORKDIR)

analyse: $(WORKDIR)
	@for f in $(SRC); do \
		echo "[GHDL -a] $$f"; \
		$(GHDL) -a $(STD) --workdir=$(WORKDIR) $$f || exit 1; \
	done

elab: analyse
	@echo "[GHDL -e] $(TB_TOP)"
	$(GHDL) -e $(STD) --workdir=$(WORKDIR) $(TB_TOP)

sim: elab
	@echo "[GHDL -r] $(TB_TOP)"
	$(GHDL) -r $(STD) --workdir=$(WORKDIR) $(TB_TOP) \
		--wave=$(WAVE_FILE) --stop-time=100us

wave: sim
	$(GTKWAVE) $(WAVE_FILE) &

clean:
	rm -rf $(WORKDIR)
	rm -f *.cf *.o *.vcd *.ghw
	rm -f mha_out.txt ffn_out.txt encoder_out.txt
