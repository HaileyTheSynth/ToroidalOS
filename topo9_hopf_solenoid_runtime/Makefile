CC       := gcc
AS       := gcc
LD       := ld

CFLAGS   := -m32 -ffreestanding -fno-stack-protector -fno-pic -fno-pie -O2 -Wall -Wextra
LDFLAGS  := -m elf_i386 -T linker.ld -nostdlib

SRC_DIR  := src
BUILD    := build
ISO_DIR  := iso
KERNEL   := $(BUILD)/kernel.elf
ISO      := $(BUILD)/topo9_hopf_solenoid.iso

.PHONY: all run clean check-tools

all: check-tools $(ISO)

check-tools:
	@command -v $(CC) >/dev/null || (echo "gcc missing"; exit 1)
	@command -v $(LD) >/dev/null || (echo "ld missing"; exit 1)
	@command -v grub-mkrescue >/dev/null || (echo "grub-mkrescue missing (install grub-pc-bin + xorriso)"; exit 1)
	@command -v qemu-system-i386 >/dev/null || (echo "qemu-system-i386 missing"; exit 1)

$(BUILD):
	mkdir -p $(BUILD)

$(KERNEL): $(BUILD) $(SRC_DIR)/boot.s $(SRC_DIR)/kernel.c linker.ld
	$(AS) -m32 -c $(SRC_DIR)/boot.s -o $(BUILD)/boot.o
	$(CC) $(CFLAGS) -c $(SRC_DIR)/kernel.c -o $(BUILD)/kernel.o
	$(LD) $(LDFLAGS) -o $(KERNEL) $(BUILD)/boot.o $(BUILD)/kernel.o

$(ISO): $(KERNEL) $(ISO_DIR)/boot/grub/grub.cfg
	mkdir -p $(ISO_DIR)/boot
	cp $(KERNEL) $(ISO_DIR)/boot/kernel.elf
	grub-mkrescue -o $(ISO) $(ISO_DIR) >/dev/null

run: $(ISO)
	qemu-system-i386 -cdrom $(ISO) -m 64M -serial stdio

clean:
	rm -rf $(BUILD) $(ISO_DIR)/boot/kernel.elf
