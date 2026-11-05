cd [file dirname [file normalize [info script]]]
set new_project [expr {![file exists eye_top.xise]}]
if {[file exists eye_top.xise]} {project open eye_top.xise} else {
    project new eye_top.xise
}
project set family Spartan6
project set device xc6slx9
project set package ftg256
project set speed -2
project set top_level_module_type HDL
project set synthesis_tool {XST (VHDL/Verilog)}
project set simulator {ISim (VHDL/Verilog)}
project set {Preferred Language} Verilog
foreach source {activation_store activity_status binocular_timer blink_event board_display cnn_addresses command_controller eye_cnn eye_cnn_parallel eye_top key_filter latency_meter mac_int8 model_store omz_cnn packet_rx packet_tx requantize_q24 uart_rx uart_tx} {
    if {![file exists rtl/$source.v]} {error "Missing rtl/$source.v"}
    if {$new_project} {xfile add rtl/$source.v}
}
foreach source {model.hex omz_biases.hex omz_weights.hex omz_weights0.hex omz_weights1.hex omz_weights2.hex} {
    if {![file exists generated/$source]} {error "Missing generated/$source"}
    if {$new_project} {xfile add generated/$source}
}
foreach source {rtl/crc16_function.vh generated/model_config.vh generated/omz_config.vh} {
    if {![file exists $source]} {error "Missing $source"}
}
if {$new_project} {xfile add board_pins.ucf}
project set top eye_top
project set {Unused IOB Pins} Float -process {Generate Programming File}
project set {FPGA Start-Up Clock} {JTAG Clock} -process {Generate Programming File}
project set {UserID Code (8 Digit Hexadecimal)} 41454545 -process {Generate Programming File}
project clean
if {![process run {Check Syntax}]} {project close; error {Syntax check failed}}
if {![process run {Synthesize - XST}]} {project close; error {Synthesis failed}}
project close
puts {SYNTHESIS PASS: pretrained OMZ P2, XC6SLX9-2FTG256, 50 MHz, 1000000 baud.}
