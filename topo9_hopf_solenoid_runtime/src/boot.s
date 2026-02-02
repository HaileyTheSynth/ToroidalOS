/* Multiboot v1 header + entry stub (32-bit) */
.set ALIGN,    1<<0
.set MEMINFO,  1<<1
.set FLAGS,    ALIGN | MEMINFO
.set MAGIC,    0x1BADB002
.set CHECKSUM, -(MAGIC + FLAGS)

.section .multiboot
.align 4
.long MAGIC
.long FLAGS
.long CHECKSUM

.section .text
.global _start
.type _start, @function
_start:
    cli
    mov $stack_top, %esp
    mov %esp, %ebp
    call kernel_main
.hang:
    hlt
    jmp .hang

.section .bss
.align 16
stack_bottom:
.skip 32768
stack_top:
