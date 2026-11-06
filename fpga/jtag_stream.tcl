# Persistent USER1 session for the isolated JTG1 experimental image only.
# Discovery/IDCODE benchmarks use separate read-only scripts, never this entrypoint.
set mode [lindex $argv 0]
if {!(($argc==1 && $mode in {experimental-probe discover}) || ($argc in {3 4} && $mode eq "program-probe"))} {error {Explicit mode required}}
connect -url tcp:127.0.0.1:3121
jtag targets -set -filter {name == "xc6slx9"} -timeout 5
set devices {}
set inventory [jtag targets -target-properties]
puts "JTINVENTORY $inventory";flush stdout
# Enumerate first: a fresh server connection may not have populated is_fpga
# properties when a property filter is evaluated. Match the proven SVF workflow.
foreach item $inventory {
    if {[dict exists $item is_fpga] && [dict get $item is_fpga]} {lappend devices $item}
}
if {[llength $devices]!=1} {error {Expected exactly one FPGA}}
set target [lindex $devices 0]
if {[dict get $target name] ne "xc6slx9" || [dict get $target idcode] ne "24001093" || [dict get $target irlen]!=6} {
    error {Wrong target; USER scan refused}
}
jtag targets -set -filter {name == "xc6slx9"}
set original_frequency [jtag frequency]
set frequencies [jtag frequency -list]
if {[lindex $argv 0] eq "discover"} {
    puts "DISCOVERY_PASS $target TCK=$original_frequency AVAILABLE=$frequencies"
    disconnect
    exit
}
jtag lock 10000
if {$mode eq "program-probe"} {
    # Explicit CLI programming stage only: the host has verified build/timing,
    # hashes and rollback. Keep the selected physical target in this session.
    set expected_usercode [lindex $argv 2]
    if {$expected_usercode ni {1247037233 1247039025}} {error {Unsupported experimental image code}}
    source [file join [file dirname [info script]] play_ise_svf.tcl]
    play_ise_svf [lindex $argv 1]
    set s [jtag sequence]
    $s irshift -state IDLE -capture -integer 6 8
    $s drshift -state IDLE -capture -tdi 0 32
    set values [$s run -integer];$s delete
    if {([lindex $values 0]&32)==0 || [lindex $values 1]!=$expected_usercode} {error {Experimental DONE/USERCODE failed}}
    puts "JTPROGRAM_PASS VOLATILE $expected_usercode";flush stdout
}
proc select_user {} {
    set s [jtag sequence]
    $s irshift -state IDLE -integer 6 2
    $s state IDLE 16
    $s run
    $s delete
}
select_user
set rollback_pending [expr {$mode eq "program-probe" && $argc==4}]
proc restore_baseline {path} {
    play_ise_svf $path
    set s [jtag sequence]
    $s irshift -state IDLE -capture -integer 6 8
    $s drshift -state IDLE -capture -tdi 0 32
    set values [$s run -integer];$s delete
    if {([lindex $values 0]&32)==0 || [lindex $values 1]!=0x41454545} {error {Baseline restoration DONE/USERCODE failed}}
}
puts "JTREADY\t$original_frequency";flush stdout
while {[gets stdin line]>=0} {
    set fields [split $line "\t"]
    set op [lindex $fields 0];set id [lindex $fields 1]
    if {$op eq "CLOSE"} {break}
    if {[catch {
        if {$op eq "RESTORE"} {
            if {!$rollback_pending} {error {No prepared baseline rollback in this session}}
            restore_baseline [lindex $argv 3]
            set rollback_pending 0
            puts "JTRESTORED\t$id"
        } elseif {$op eq "FREQUENCY"} {
            set frequency [lindex $fields 2]
            if {[lsearch -exact $frequencies $frequency]<0} {error {Unsupported TCK}}
            puts "JTFREQ\t$id\t[jtag frequency $frequency]"
        } elseif {$op eq "RESET"} {
            set s [jtag sequence];$s state RESET 16;$s state IDLE 16;$s run;$s delete
            select_user
            puts "JTRESET\t$id"
        } elseif {$op eq "SCAN"} {
            set capture [lindex $fields 2];set count [lindex $fields 3];set hex [lindex $fields 4]
            if {$capture ni {0 1} || ![string is integer -strict $count] || $count<1 || $count>16 ||
                ![regexp {^[0-9a-fA-F]+$} $hex] || [string length $hex]%2 || [string length $hex]>131072} {
                error {Invalid bounded SCAN arguments}
            }
            set before [clock microseconds]
            set data [binary format H* $hex]
            set bits [expr {[string length $data]*8}]
            set s [jtag sequence]
            for {set i 0} {$i<$count} {incr i} {
                if {$capture} {$s drshift -state IDLE -capture -binary $bits $data} else {$s drshift -state IDLE -binary $bits $data}
            }
            set built [clock microseconds]
            set values [$s run -binary]
            set executed [clock microseconds]
            $s delete
            set result ""
            foreach value $values {append result $value}
            binary scan $result H* outhex
            set formatted [clock microseconds]
            puts "JTSCAN\t$id\t[expr {$built-$before}]\t[expr {$executed-$built}]\t[expr {$formatted-$executed}]\t$outhex"
        } else {error {Unknown operation}}
    } message options]} {
        binary scan [encoding convertto utf-8 $message] H* errhex
        puts "JTERROR\t$id\t$errhex"
    }
    flush stdout
}
if {$rollback_pending} {
    restore_baseline [lindex $argv 3]
    puts {JTAUTO_RESTORE_PASS};flush stdout
}
catch {jtag frequency $original_frequency}
# Leave a documented read-only instruction before releasing the persistent owner.
# This also resets the probe FIFO; the experiment is already complete.
set s [jtag sequence]
$s state RESET 16
$s irshift -state IDLE -integer 6 9
$s drshift -state IDLE -capture -tdi 0 32
set final_id [lindex [$s run -integer] 0];$s delete
if {$final_id!=0x24001093} {error {Final target identity mismatch}}
puts {JTCLOSE_PASS IDCODE_VERIFIED};flush stdout
catch {jtag unlock}
disconnect
exit
