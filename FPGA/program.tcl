set directory [file dirname [file normalize [info script]]]
proc main {directory} {
    cd $directory
    foreach source [concat [glob -nocomplain rtl/*.v rtl/*.vh generated/*.hex generated/*.vh] {board_pins.ucf create_project.tcl build.tcl}] {
        if {[file mtime $source] > [file mtime eye_top.bit]} {error "Stale bitstream: $source changed; run build.cmd before programming"}
    }
    if {[file mtime eye_top.svf] < [file mtime eye_top.bit]} {error {Stale SVF: run build.cmd}}
    connect -url tcp:127.0.0.1:3121
    set devices {}
    foreach target [jtag targets -target-properties] {
        if {[dict exists $target is_fpga] && [dict get $target is_fpga]} {lappend devices $target}
    }
    if {[llength $devices]!=1 || [dict get [lindex $devices 0] name] ne "xc6slx9"} {error {Expected one XC6SLX9}}
    jtag targets -set -filter {name == "xc6slx9"}
    jtag lock 10000
    source {play_ise_svf.tcl}
    play_ise_svf eye_top.svf
    set sequence [jtag sequence]
    $sequence state RESET
    $sequence irshift -state IDLE -capture -integer 6 8
    $sequence drshift -state IDLE -capture -tdi 0 32
    set result [$sequence run -integer]
    $sequence delete
    puts [format {IR status: 0x%02X; USERCODE: 0x%08X} [lindex $result 0] [lindex $result 1]]
    if {([lindex $result 0]&32)==0 || [lindex $result 1]!=0x41454545} {error {DONE/USERCODE verification failed}}
    jtag unlock
    puts {PASS: actual CNN/UART bitstream loaded; DONE high; USERCODE AEEE.}
}
if {[catch {main $directory} message options]} {catch {jtag unlock};puts stderr [dict get $options -errorinfo];exit 1}
exit 0
