module uart_tx #(parameter integer CLOCKS_PER_BIT=434)(input wire clk, input wire reset,
    input wire start, input wire [7:0] data, output reg tx=1, output reg busy=0);
    reg [9:0] shift=10'h3ff;
    reg [3:0] bit_index=0;
    reg [15:0] count=0;
    always @(posedge clk) begin
        if (reset) begin busy<=0; tx<=1; end
        else if (start && !busy) begin
            shift<={1'b1,data,1'b0}; tx<=0; busy<=1;
            bit_index<=0; count<=CLOCKS_PER_BIT-1;
        end else if (busy) begin
            if (count!=0) count<=count-1'b1;
            else if (bit_index==9) begin busy<=0; tx<=1; end
            else begin
                shift<={1'b1,shift[9:1]}; tx<=shift[1]; bit_index<=bit_index+1'b1;
                count<=CLOCKS_PER_BIT-1;
            end
        end
    end
endmodule
