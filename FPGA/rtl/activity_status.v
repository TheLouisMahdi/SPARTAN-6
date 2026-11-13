// Per-eye transition pulses and completed inference pairs per one-second window.
module activity_status #(parameter integer CLOCK_HZ=50000000)(
    input wire clk, input wire reset, input wire update,
    input wire result_valid, input wire [1:0] closed,
    output wire [1:0] pulse, output reg [7:0] fps_bcd=0);
    localparam integer PULSE_CYCLES=CLOCK_HZ*3/25; // 120 ms
    reg [31:0] second_ticks=0;
    reg [31:0] left_ticks=0,right_ticks=0;
    reg [7:0] count_bcd=0;
    reg [1:0] previous=0;
    reg have_previous=0;
    wire [7:0] incremented=count_bcd==8'h99 ? 8'h99 :
        count_bcd[3:0]==9 ? {count_bcd[7:4]+4'd1,4'd0} : count_bcd+1'b1;
    assign pulse={right_ticks!=0,left_ticks!=0};
    always @(posedge clk) begin
        if(reset)begin
            second_ticks<=0;count_bcd<=0;fps_bcd<=0;
            left_ticks<=0;right_ticks<=0;previous<=0;have_previous<=0;
        end else begin
            if(second_ticks==CLOCK_HZ-1)begin
                second_ticks<=0;fps_bcd<=update ? incremented : count_bcd;count_bcd<=0;
            end else begin
                second_ticks<=second_ticks+1'b1;
                if(update)count_bcd<=incremented;
            end
            if(left_ticks!=0)left_ticks<=left_ticks-1'b1;
            if(right_ticks!=0)right_ticks<=right_ticks-1'b1;
            if(update)begin
                if(have_previous && closed[0]!=previous[0])left_ticks<=PULSE_CYCLES;
                if(have_previous && closed[1]!=previous[1])right_ticks<=PULSE_CYCLES;
                previous<=closed;have_previous<=1;
            end else if(!result_valid)begin
                have_previous<=0;left_ticks<=0;right_ticks<=0;
            end
        end
    end
endmodule
