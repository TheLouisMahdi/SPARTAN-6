// Measure a two-eye FPGA transaction in milliseconds, rounded and capped at 999.
module latency_meter #(parameter integer CLOCK_HZ=50000000)(
    input wire clk, input wire reset, input wire start,
    input wire active, input wire finish, output reg [11:0] milliseconds_bcd=0);
    localparam integer TICKS_PER_MS=CLOCK_HZ/1000;
    reg [31:0] ticks=0;
    reg [11:0] elapsed_bcd=0;
    function [11:0] increment_bcd;
        input [11:0] value;
        begin
            if(value==12'h999)increment_bcd=12'h999;
            else if(value[3:0]!=9)increment_bcd=value+1'b1;
            else if(value[7:4]!=9)increment_bcd={value[11:8],value[7:4]+4'd1,4'd0};
            else increment_bcd={value[11:8]+4'd1,8'd0};
        end
    endfunction
    always @(posedge clk) begin
        if(reset)begin ticks<=0;elapsed_bcd<=0;milliseconds_bcd<=0;end
        else if(start)begin ticks<=0;elapsed_bcd<=0;end
        else if(active)begin
            if(ticks==TICKS_PER_MS-1)begin
                ticks<=0;elapsed_bcd<=increment_bcd(elapsed_bcd);
            end else ticks<=ticks+1'b1;
        end else if(finish)begin
            milliseconds_bcd<=ticks>=(TICKS_PER_MS+1)/2 ? increment_bcd(elapsed_bcd) : elapsed_bcd;
        end
    end
endmodule
