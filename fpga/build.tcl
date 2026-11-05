cd [file dirname [file normalize [info script]]]
project open eye_top.xise
project set {Unused IOB Pins} Float -process {Generate Programming File}
project set {FPGA Start-Up Clock} {JTAG Clock} -process {Generate Programming File}
project set {UserID Code (8 Digit Hexadecimal)} 41454545 -process {Generate Programming File}
project save
foreach step {{Implement Design} {Generate Programming File}} {
    puts "STEP: $step"
    if {![process run $step]} {project close; error "Failed: $step"}
}
project close
puts {BUILD COMPLETE: eye_top.bit}
