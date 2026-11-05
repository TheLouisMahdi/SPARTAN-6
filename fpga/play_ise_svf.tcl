# Execute the subset of SVF emitted by generate_svf.cmd for this single FPGA.
# Reject unsupported operations and preserve every TDO comparison from iMPACT.
proc svf_bytes {hex} {
    regsub -all {\s} $hex {} hex
    if {[string length $hex] % 2} {set hex "0$hex"}
    return [string reverse [binary format H* $hex]]
}
proc play_ise_svf {filename} {
    set f [open $filename r]
    set script [read $f]
    close $f
    regsub -all {//[^\n]*} $script {} script
    set endir IDLE
    set enddr IDLE
    set masks [dict create]
    foreach command [split $script {;}] {
        set command [string trim $command]
        if {$command eq {}} {continue}
        set verb [lindex $command 0]
        switch -- $verb {
            TRST {if {$command ne {TRST OFF}} {error "Unsupported SVF: $command"}}
            ENDIR {set endir [lindex $command 1]}
            ENDDR {set enddr [lindex $command 1]}
            FREQUENCY {
                jtag frequency [expr {int([lindex $command 1])}]
            }
            TIR - HIR - TDR - HDR {
                if {[lindex $command 1] != 0} {error {Only a single-device SVF is supported.}}
            }
            STATE {
                set seq [jtag sequence]
                foreach state [lrange $command 1 end] {$seq state $state}
                $seq run
                $seq delete
            }
            RUNTEST {
                if {![regexp {^RUNTEST ([0-9]+) TCK$} $command -> count]} {
                    error "Unsupported SVF: $command"
                }
                set seq [jtag sequence]
                $seq state IDLE $count
                $seq run
                $seq delete
            }
            SIR - SDR {
                set bits [lindex $command 1]
                if {![regexp {TDI\s+\(([0-9a-fA-F\s]+)\)} $command -> hex]} {
                    error {Missing SVF TDI data.}
                }
                set seq [jtag sequence]
                set shift [expr {$verb eq "SIR" ? "irshift" : "drshift"}]
                set endstate [expr {$verb eq "SIR" ? $endir : $enddr}]
                set compare [regexp {TDO\s+\(([0-9a-fA-F]+)\)} $command -> expected]
                if {$compare} {
                    $seq $shift -state $endstate -capture -binary $bits [svf_bytes $hex]
                    set actual [lindex [$seq run -integer] 0]
                    if {[regexp {\mMASK\s+\(([0-9a-fA-F]+)\)} $command -> mask]} {
                        dict set masks "$verb,$bits" "0x$mask"
                    }
                    if {[dict exists $masks "$verb,$bits"]} {
                        set mask_value [dict get $masks "$verb,$bits"]
                    } else {
                        set mask_value [expr {(1 << $bits) - 1}]
                    }
                    set expected_value "0x$expected"
                    if {($actual & $mask_value) != ($expected_value & $mask_value)} {
                        error [format {SVF TDO mismatch: got 0x%X, expected %s, mask 0x%X} $actual $expected $mask_value]
                    }
                } else {
                    $seq $shift -state $endstate -binary $bits [svf_bytes $hex]
                    $seq run
                }
                $seq delete
            }
            default {error "Unsupported SVF command: $verb"}
        }
    }
    puts {ISE/iMPACT SVF programming sequence completed.}
}
