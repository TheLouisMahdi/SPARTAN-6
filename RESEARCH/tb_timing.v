`timescale 1ns/1ps
`include "suite.vh"
module tb_timing;
    reg clk=0,reset=0,invalidate=0,update=0;
    reg [31:0] timestamp_ms=0;
    reg [1:0] valid_mask=0;
    reg [15:0] left_score=0,right_score=0;
    wire [1:0] closed_mask,known_mask;
    wire event_pulse;
    wire [31:0] count,long_count,left_current,right_current,last_left,last_right,last_overlap,last_duration;
    reg [66:0] vectors[0:`SAMPLES-1];
    reg [260:0] expected[0:`SAMPLES-1];
    wire [260:0] actual={event_pulse,known_mask,closed_mask,count,long_count,left_current,right_current,
                        last_left,last_right,last_overlap,last_duration};
    integer i;
    binocular_timer dut(clk,reset,invalidate,update,timestamp_ms,valid_mask,left_score,right_score,
        closed_mask,known_mask,event_pulse,count,long_count,left_current,right_current,
        last_left,last_right,last_overlap,last_duration);
    always #5 clk=~clk;
    initial begin
        $readmemh("vectors.hex",vectors);$readmemh("expected.hex",expected);
        for(i=0;i<`SAMPLES;i=i+1)begin
            @(negedge clk);
            if(vectors[i][66])begin reset=1;@(negedge clk);reset=0;end
            {timestamp_ms,valid_mask,left_score,right_score}=vectors[i][65:0];
            update=1;@(posedge clk);#1;
            if(actual!==expected[i])begin
                $display("TIMING_FAIL sample=%0d actual=%h expected=%h",i,actual,expected[i]);$finish;
            end
            @(negedge clk);update=0;@(posedge clk);#1;
            if(event_pulse)begin $display("TIMING_FAIL event pulse width");$finish;end
        end
        $display("TIMING_PASS samples=%0d",`SAMPLES);$finish;
    end
    initial begin #1000000;$display("TIMING_FAIL timeout");$finish;end
endmodule
